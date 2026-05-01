"""Autoregressive STAR-MD trajectory generation.

Generates protein MD trajectories by rolling out the conditional model
one frame at a time using reverse SE(3) diffusion.

Training scheme (Diffusion Forcing) uses tau ~ U[0, max_t] with max_t ≈ 0.1,
so context frames are always approximately clean.  KV-cache is used for
efficient inference: context frames are cached once, and only the target frame
is processed through the model per denoising step.

Usage:
    python gen_model/inference_conditional.py \
        --checkpoint checkpoints/conditional/last.ckpt \
        --data_dir data \
        --output traj_generated.pt \
        --total_frames 100 \
        --delta_t 0.1

Output:
    PyTorch tensor [T, N, 7] of generated rigid frames (backbone, scaled),
    saved to --output via torch.save().
"""

import argparse
import os
import time

import numpy as np
import torch

import gen_model.path_setup  # noqa: F401
from gen_model.utils.rigid_utils import Rigid

from gen_model.train_base import default_se3_conf, default_model_conf
from gen_model.models.star_score_network import StarScoreNetwork
from gen_model.models.lora import apply_lora
from gen_model.diffusion.se3_diffuser import SE3Diffuser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(x, dtype=dtype)


def compute_coord_scale(npy_path: str, num_samples: int = 500) -> float:
    """Compute coordinate scale factor (1/std) from a trajectory .npy file.

    Used by both conditional and unconditional inference to match the
    training-time normalisation.
    """
    arr = np.load(npy_path)
    num_frames = arr.shape[0]
    sample_idx = np.random.choice(num_frames, min(num_samples, num_frames), replace=False)
    ca_coords = []
    for fi in sample_idx:
        ca = arr[fi, :, 1, :].astype(np.float32)  # atom14 index 1 = CA
        ca = ca - np.mean(ca, axis=0, keepdims=True)
        ca_coords.append(ca)
    all_ca = np.concatenate(ca_coords, axis=0)
    std = np.std(all_ca)
    return float(1.0 / (std + 1e-8))


def _noise_rigids(rigids7: torch.Tensor, t: float, diffuser, mask_np: np.ndarray) -> torch.Tensor:
    """Apply SE3 forward marginal at level t to a [N, 7] rigid tensor."""
    device = rigids7.device
    r = Rigid.from_tensor_7(rigids7.cpu().float())
    diff = diffuser.forward_marginal(
        rigids_0=r, t=t, diffuse_mask=mask_np, as_tensor_7=True)
    return _to_tensor(diff['rigids_t']).to(device)




# ---------------------------------------------------------------------------
# Autoregressive rollout
# ---------------------------------------------------------------------------

def _make_feats_single(frame7, sc_ca, frame_abs, seq_idx, res_mask, fixed_mask,
                        delta_t, t_val, N, device):
    """Build input_feats dict for a single frame (L=1, B=1)."""
    return {
        'rigids_t':               frame7.unsqueeze(0).unsqueeze(0),        # [1,1,N,7]
        'sc_ca_t':                sc_ca.unsqueeze(0).unsqueeze(0),         # [1,1,N,3]
        'frame_idx':              torch.tensor([frame_abs], dtype=torch.long, device=device),
        'delta_t':                torch.tensor([delta_t], dtype=torch.float32, device=device),
        'res_mask':               res_mask.unsqueeze(0),                    # [1,N]
        'fixed_mask':             fixed_mask.unsqueeze(0),                  # [1,N]
        'seq_idx':                seq_idx.unsqueeze(0),                     # [1,N]
        't':                      torch.tensor([t_val], dtype=torch.float32, device=device),
        'torsion_angles_sin_cos': torch.zeros(1, 1, N, 7, 2, device=device),
    }


def _make_feats_block(rigids7_block, sc_ca_block, frame_abs_block,
                      seq_idx, res_mask, fixed_mask,
                      delta_t, t_block, N, device):
    """Build input_feats for a K-frame active block (B=1, L=K).

    Args:
        rigids7_block:    [K, N, 7]  current rigids for the K active frames.
        sc_ca_block:      [K, N, 3]  per-frame self-conditioning CA positions.
        frame_abs_block:  [K]        absolute frame indices for RoPE.
        t_block:          [1, K]     per-frame diffusion times. Per-frame so it
                                     triggers the multi-frame branch in
                                     _build_cond / _apply_se3_noise. Synchronous
                                     scheduling means all K values are equal at
                                     each denoising step, but per-frame is the
                                     general shape.
    """
    K = rigids7_block.shape[0]
    return {
        'rigids_t':               rigids7_block.unsqueeze(0),               # [1, K, N, 7]
        'sc_ca_t':                sc_ca_block.unsqueeze(0),                 # [1, K, N, 3]
        'frame_idx':              frame_abs_block,                          # [K]
        'delta_t':                torch.tensor([delta_t], dtype=torch.float32, device=device),
        'res_mask':               res_mask.unsqueeze(0),                    # [1, N]
        'fixed_mask':             fixed_mask.unsqueeze(0),                  # [1, N]
        'seq_idx':                seq_idx.unsqueeze(0),                     # [1, N]
        't':                      t_block,                                  # [1, K]
        'torsion_angles_sin_cos': torch.zeros(1, K, N, 7, 2, device=device),
    }


def _trim_kv_caches(kv_caches: list, max_frames: int, N: int):
    """Trim each cache to keep at most max_frames * N tokens (oldest dropped)."""
    max_tokens = max_frames * N
    for b in range(len(kv_caches)):
        c = kv_caches[b]
        if c is not None and c['k'].shape[2] > max_tokens:
            kv_caches[b] = {
                'k': c['k'][:, :, -max_tokens:, :].contiguous(),
                'v': c['v'][:, :, -max_tokens:, :].contiguous(),
            }


@torch.no_grad()
def rollout(
    model: StarScoreNetwork,
    diffuser,
    seed_rigids: torch.Tensor,
    seq_idx: torch.Tensor,
    res_mask: torch.Tensor,
    total_frames: int,
    num_context: int,
    delta_t: float,
    n_steps: int,
    min_t: float,
    max_t: float,
    device: str,
) -> list:
    """KV-cached autoregressive rollout.

    Context frames are processed once to populate per-block K/V caches.
    Each denoising step for the target frame runs through the model with
    L=1 (target only) + cached context K/V, rather than re-encoding the
    full window at every step.

    Cost vs. uncached: (L-1) cache-init passes per frame (done once)
    + n_steps × 1-frame passes (vs. n_steps × L-frame passes before).
    For L=8, n_steps=100: ~8× cheaper per target frame.

    Cache is trimmed to keep the most recent (num_context - 1) frames,
    matching the training window size.
    """
    N          = seed_rigids.shape[0]
    num_blocks = model.score_model._ipa_conf.num_blocks
    mask_np    = res_mask.cpu().numpy()
    fixed_mask = torch.zeros_like(res_mask)

    history_clean = [seed_rigids.to(device)]
    kv_caches     = [None] * num_blocks   # persistent K/V per block

    # Seed frame → initialise cache (t ≈ min_t, near-clean)
    sc_ca_seed = torch.zeros(N, 3, device=device)
    seed_feats = _make_feats_single(seed_rigids, sc_ca_seed, 0,
                                    seq_idx, res_mask, fixed_mask,
                                    delta_t, min_t, N, device)
    model(seed_feats, kv_caches=kv_caches, update_kv_cache=True)

    # Performance tracking
    _flops_per_forward = None
    _frame_times       = []

    ts = np.linspace(max_t, min_t, n_steps + 1)

    for frame_num in range(1, total_frames):
        _t_start = time.perf_counter()

        frame_abs = frame_num
        sc_ca     = history_clean[-1][:, 4:7]   # translation of previous clean frame

        # Noise-initialise the target frame
        identity7 = torch.zeros(N, 7, device=device)
        identity7[:, 0] = 1.0
        target7 = _noise_rigids(identity7, max_t, diffuser, mask_np)

        # Measure FLOPs once on the first frame
        if frame_num == 1 and _flops_per_forward is None:
            try:
                import warnings
                from torch.profiler import profile, ProfilerActivity
                _probe = _make_feats_single(target7, sc_ca, frame_abs,
                                            seq_idx, res_mask, fixed_mask,
                                            delta_t, max_t, N, device)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with profile(activities=[ProfilerActivity.CUDA,
                                             ProfilerActivity.CPU],
                                 with_flops=True, record_shapes=True) as prof:
                        model(_probe, kv_caches=kv_caches)
                if device != 'cpu':
                    torch.cuda.synchronize()
                _flops_per_forward = sum(
                    e.flops for e in prof.key_averages() if e.flops > 0)
            except Exception:
                _flops_per_forward = 0

        # ── Denoising loop (L=1 target + cached context, cache read-only) ──
        for step_i in range(n_steps):
            t_now  = float(ts[step_i])
            t_next = float(ts[step_i + 1])
            dt     = t_now - t_next

            feats = _make_feats_single(target7, sc_ca, frame_abs,
                                       seq_idx, res_mask, fixed_mask,
                                       delta_t, t_now, N, device)
            pred = model(feats, kv_caches=kv_caches, update_kv_cache=False)

            rot_score   = pred['rot_score'][0, 0].cpu().numpy()
            trans_score = pred['trans_score'][0, 0].cpu().numpy()

            perturbed = diffuser.reverse(
                rigid_t=Rigid.from_tensor_7(target7.cpu().float()),
                rot_score=rot_score,
                trans_score=trans_score,
                diffuse_mask=mask_np,
                t=t_now, dt=dt,
                center=False, noise_scale=1.0,
            )
            target7 = _to_tensor(perturbed.to_tensor_7()).to(device)

        history_clean.append(target7)

        # ── Append finalized clean frame to cache (once, not per step) ──────
        cache_feats = _make_feats_single(target7, sc_ca, frame_abs,
                                          seq_idx, res_mask, fixed_mask,
                                          delta_t, min_t, N, device)
        model(cache_feats, kv_caches=kv_caches, update_kv_cache=True)

        # Trim cache to training window size
        _trim_kv_caches(kv_caches, max_frames=num_context - 1, N=N)

        if device != 'cpu':
            torch.cuda.synchronize()
        _frame_times.append(time.perf_counter() - _t_start)

        if frame_num % 10 == 0:
            avg_s = float(np.mean(_frame_times))
            if _flops_per_forward and _flops_per_forward > 0:
                tflops_frame = (_flops_per_forward * n_steps) / 1e12
                print(f'  frame {frame_num}/{total_frames} | '
                      f'{avg_s:.2f}s/frame | '
                      f'{tflops_frame:.3f} TFLOP/frame | '
                      f'{tflops_frame / avg_s:.2f} TFLOP/s')
            else:
                print(f'  frame {frame_num}/{total_frames} | {avg_s:.2f}s/frame')

    if _frame_times:
        avg_s   = float(np.mean(_frame_times))
        total_s = float(np.sum(_frame_times))
        print(f'\nPerformance summary:')
        print(f'  Frames generated : {len(_frame_times)}')
        print(f'  Total wall time  : {total_s:.1f}s')
        print(f'  Avg per frame    : {avg_s:.2f}s')
        if _flops_per_forward and _flops_per_forward > 0:
            tflops_frame = (_flops_per_forward * n_steps) / 1e12
            print(f'  TFLOP/frame      : {tflops_frame:.3f}')
            print(f'  Avg TFLOP/s      : {tflops_frame / avg_s:.2f}')

    return history_clean


# ---------------------------------------------------------------------------
# Block-AR rollout (K-frame joint denoising per iteration)
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_block_ar(
    model: StarScoreNetwork,
    diffuser,
    seed_rigids: torch.Tensor,
    seq_idx: torch.Tensor,
    res_mask: torch.Tensor,
    total_frames: int,
    num_context: int,
    block_size: int,
    delta_t: float,
    n_steps: int,
    min_t: float,
    max_t: float,
    device: str,
) -> list:
    """Block-AR rollout: jointly denoise K new frames per iteration.

    Compared to the K=1 `rollout()` above:
      - K = block_size frames are denoised together with full intra-block
        bidirectional information flow (each active frame attends to earlier
        active frames + clean history). This is what gives block-AR its
        intra-block coherence vs. pure AR.
      - After n_steps the K finalized frames are pushed into the persistent
        KV cache in one update_kv_cache pass and the window advances by K.
      - All K active frames are at the same t at each denoising step
        (synchronous block). The trained model also handles per-frame
        independent t (Diffusion Forcing), so a staggered/pyramid schedule
        is in-distribution and a natural follow-up — left out here for
        simplicity.

    Self-conditioning (sc_ca_t): all K active frames condition on the most
    recent finalized frame's CA. Per-frame self-conditioning across the
    block would require carrying intermediate predictions step-to-step;
    the current model trained without that, so the simple choice is faithful.

    Reduces to the existing K=1 path when block_size == 1.
    """
    assert block_size >= 1
    N          = seed_rigids.shape[0]
    K          = block_size
    num_blocks = model.score_model._ipa_conf.num_blocks
    mask_np    = res_mask.cpu().numpy()
    fixed_mask = torch.zeros_like(res_mask)

    history_clean = [seed_rigids.to(device)]
    kv_caches     = [None] * num_blocks

    # Seed the cache with the initial frame at near-zero t (one-frame block).
    sc_ca_seed = torch.zeros(N, 3, device=device)
    seed_feats = _make_feats_block(
        rigids7_block=seed_rigids.unsqueeze(0),                     # [1, N, 7]
        sc_ca_block=sc_ca_seed.unsqueeze(0),                        # [1, N, 3]
        frame_abs_block=torch.tensor([0], dtype=torch.long, device=device),
        seq_idx=seq_idx, res_mask=res_mask, fixed_mask=fixed_mask,
        delta_t=delta_t,
        t_block=torch.full((1, 1), min_t, device=device),           # [1, 1]
        N=N, device=device,
    )
    model(seed_feats, kv_caches=kv_caches, update_kv_cache=True)

    ts = np.linspace(max_t, min_t, n_steps + 1)
    frames_produced = 1   # seed counts as produced

    _block_times = []

    while frames_produced < total_frames:
        K_now = min(K, total_frames - frames_produced)
        _t_start = time.perf_counter()

        # Initialise K_now pure-noise frames at t=max_t
        identity7 = torch.zeros(N, 7, device=device)
        identity7[:, 0] = 1.0
        active = torch.stack(
            [_noise_rigids(identity7, max_t, diffuser, mask_np)
             for _ in range(K_now)],
            dim=0,
        )                                                            # [K_now, N, 7]

        sc_ca_active = history_clean[-1][:, 4:7].unsqueeze(0).expand(K_now, N, 3).contiguous()
        frame_abs    = torch.arange(
            frames_produced, frames_produced + K_now,
            dtype=torch.long, device=device,
        )

        # Joint denoising with read-only KV cache
        for step_i in range(n_steps):
            t_now  = float(ts[step_i])
            t_next = float(ts[step_i + 1])
            dt     = t_now - t_next
            t_block = torch.full((1, K_now), t_now, device=device)

            feats = _make_feats_block(
                rigids7_block=active, sc_ca_block=sc_ca_active,
                frame_abs_block=frame_abs,
                seq_idx=seq_idx, res_mask=res_mask, fixed_mask=fixed_mask,
                delta_t=delta_t, t_block=t_block, N=N, device=device,
            )
            pred = model(feats, kv_caches=kv_caches, update_kv_cache=False)

            # Per-frame reverse step (each active frame is independent in this op)
            new_active = []
            for l in range(K_now):
                rot_score   = pred['rot_score'][0, l].cpu().numpy()
                trans_score = pred['trans_score'][0, l].cpu().numpy()
                perturbed = diffuser.reverse(
                    rigid_t=Rigid.from_tensor_7(active[l].cpu().float()),
                    rot_score=rot_score, trans_score=trans_score,
                    diffuse_mask=mask_np,
                    t=t_now, dt=dt,
                    center=False, noise_scale=1.0,
                )
                new_active.append(_to_tensor(perturbed.to_tensor_7()).to(device))
            active = torch.stack(new_active, dim=0)

        # Finalise the block: append to history, push into persistent KV cache
        for l in range(K_now):
            history_clean.append(active[l])

        cache_t = torch.full((1, K_now), min_t, device=device)
        cache_feats = _make_feats_block(
            rigids7_block=active, sc_ca_block=sc_ca_active,
            frame_abs_block=frame_abs,
            seq_idx=seq_idx, res_mask=res_mask, fixed_mask=fixed_mask,
            delta_t=delta_t, t_block=cache_t, N=N, device=device,
        )
        model(cache_feats, kv_caches=kv_caches, update_kv_cache=True)

        # Trim cache to the training window so attention stays in-distribution.
        _trim_kv_caches(kv_caches, max_frames=num_context - 1, N=N)

        if device != 'cpu':
            torch.cuda.synchronize()
        _block_times.append(time.perf_counter() - _t_start)
        frames_produced += K_now

        if (frames_produced - 1) % max(10, K) == 0 or frames_produced == total_frames:
            avg_s = float(np.mean(_block_times))
            print(f'  produced {frames_produced}/{total_frames} | '
                  f'{avg_s:.2f}s/block (K={K_now})')

    if _block_times:
        avg_block_s = float(np.mean(_block_times))
        total_s     = float(np.sum(_block_times))
        per_frame_s = total_s / max(1, frames_produced - 1)
        print(f'\nPerformance summary (block-AR, K={K}):')
        print(f'  Frames generated : {frames_produced - 1}')
        print(f'  Total wall time  : {total_s:.1f}s')
        print(f'  Avg per block    : {avg_block_s:.2f}s')
        print(f'  Avg per frame    : {per_frame_s:.2f}s')

    return history_clean


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='STAR-MD autoregressive rollout')
    parser.add_argument('--checkpoint',    type=str, required=True)
    parser.add_argument('--data_dir',      type=str, default='data')
    parser.add_argument('--atlas_csv',     type=str, default='gen_model/splits/atlas.csv')
    parser.add_argument('--suffix',        type=str, default=None,
                        help='File suffix for trajectory .npy files (e.g. "_latent"). '
                             'Auto-detected from data_dir if not specified.')
    parser.add_argument('--protein',       type=str, default=None,
                        help='Protein name to seed from (uses first val frame)')
    parser.add_argument('--output',        type=str, default='generated_traj.pt')
    parser.add_argument('--total_frames',  type=int, default=250,
                        help='Number of frames to generate (ATLAS default: 250)')
    parser.add_argument('--num_frames',    type=int, default=8,
                        help='Training window size L (must match checkpoint)')
    parser.add_argument('--delta_t',       type=float, default=0.1,
                        help='Physical stride between frames (ns)')
    parser.add_argument('--n_steps',       type=int, default=150)
    parser.add_argument('--min_t',         type=float, default=0.01)
    parser.add_argument('--max_t',         type=float, default=0.1)
    parser.add_argument('--block_size',    type=int, default=1,
                        help='Block-AR block size K. K=1 (default) is the existing '
                             'pure-AR rollout; K>1 jointly denoises K frames per '
                             'iteration (requires Diffusion Forcing per-frame t '
                             'training, available after the per-frame t change).')
    parser.add_argument('--coord_scale',   type=float, default=0.1,
                        help='Coordinate scale used during training (default 0.1 per paper)')
    parser.add_argument('--lora_r',        type=int,   default=0)
    parser.add_argument('--lora_alpha',    type=float, default=0.0)
    parser.add_argument('--st_num_heads',  type=int,   default=8)
    parser.add_argument('--num_blocks',    type=int,   default=8,
                        help='Number of IPA blocks (must match checkpoint)')
    parser.add_argument('--spatial_sigma', type=float, default=0.0,
                        help='Spatial Gaussian bias in ST attention (must match training)')
    parser.add_argument('--device',        type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device

    # Build model
    se3_conf   = default_se3_conf()
    model_conf = default_model_conf(
        star_enabled=True,
        st_num_heads=args.st_num_heads,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_blocks=args.num_blocks,
        spatial_sigma=args.spatial_sigma,
    )
    diffuser      = SE3Diffuser(se3_conf)
    score_network = StarScoreNetwork(model_conf, diffuser)
    apply_lora(score_network, model_conf.lora)

    # Load checkpoint — prefer EMA shadow weights when available.
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'ema_shadow' in ckpt:
        # EMA shadow keys are already unprefixed (relative to self.model)
        state = {k: v for k, v in ckpt['ema_shadow'].items()
                 if not k.endswith('.rope.inv_freq')}
        print('Loading EMA weights from checkpoint.')
    else:
        state = ckpt.get('state_dict', ckpt)
        state = {k[len('model.'):]: v for k, v in state.items()
                 if k.startswith('model.')}
        state = {k: v for k, v in state.items()
                 if not k.endswith('.rope.inv_freq')}
        print('Warning: no EMA weights in checkpoint, using raw state_dict.')
    score_network.load_state_dict(state, strict=False)
    score_network.eval().to(device)

    # Seed frame from first val sample
    import glob as _glob
    import pandas as pd
    from gen_model.data.geometry import atom14_to_frames, atom14_to_atom37
    from gen_model.utils.rigid_utils import Rigid, Rotation

    # --- Find trajectory file directly (no split CSV needed) ---
    prot_dir   = os.path.join(args.data_dir, args.protein)
    candidates = sorted(_glob.glob(os.path.join(prot_dir, f'{args.protein}_R1*.npy')))
    if not candidates:
        raise FileNotFoundError(f'No trajectory found for {args.protein} in {prot_dir}')
    npy_path = candidates[0]
    print(f'Seed trajectory: {npy_path}')

    # --- Load sequence from atlas.csv ---
    atlas_df = pd.read_csv(args.atlas_csv, index_col='name')
    seqres   = atlas_df.loc[args.protein, 'seqres']
    from gen_model.data.residue_constants import restype_order
    aatype_np = np.array([restype_order[c] for c in seqres])
    N_res     = len(aatype_np)

    # --- Load first frame and compute rigid ---
    arr        = np.load(npy_path)          # [T, N, 14, 3]
    frame0     = arr[0:1].astype(np.float32)  # [1, N, 14, 3]
    aatype_t   = torch.from_numpy(aatype_np)[None]          # [1, N]
    frames_out = atom14_to_frames(torch.from_numpy(frame0))
    rigid0     = Rigid(
        rots=Rotation(rot_mats=frames_out._rots._rot_mats[0]),
        trans=frames_out._trans[0],
    )
    r7 = rigid0.to_tensor_7()
    # Centre and scale translations to match training coord_scale.
    ca0         = torch.from_numpy(frame0[0, :, 1, :])   # [N, 3]
    centroid    = ca0.mean(dim=0, keepdim=True)
    coord_scale = args.coord_scale
    r7[..., 4:] = (r7[..., 4:] - centroid) * coord_scale
    print(f'coord_scale : {coord_scale} (--coord_scale)')

    seed_rigids = r7.to(device)                              # [N, 7]
    seq_idx     = torch.arange(1, N_res + 1, device=device) # [N]
    res_mask    = torch.ones(N_res, device=device)           # [N]

    print(f'Generating {args.total_frames} frames '
          f'(N={seq_idx.shape[0]} residues, delta_t={args.delta_t} ns) ...')

    if args.block_size == 1:
        generated = rollout(
            model=score_network,
            diffuser=diffuser,
            seed_rigids=seed_rigids,
            seq_idx=seq_idx,
            res_mask=res_mask,
            total_frames=args.total_frames,
            num_context=args.num_frames,
            delta_t=args.delta_t,
            n_steps=args.n_steps,
            min_t=args.min_t,
            max_t=args.max_t,
            device=device,
        )
    else:
        print(f'Block-AR rollout: K={args.block_size}')
        generated = rollout_block_ar(
            model=score_network,
            diffuser=diffuser,
            seed_rigids=seed_rigids,
            seq_idx=seq_idx,
            res_mask=res_mask,
            total_frames=args.total_frames,
            num_context=args.num_frames,
            block_size=args.block_size,
            delta_t=args.delta_t,
            n_steps=args.n_steps,
            min_t=args.min_t,
            max_t=args.max_t,
            device=device,
        )

    traj = torch.stack(generated, dim=0).cpu()   # [T, N, 7]
    torch.save(traj, args.output)
    print(f'Saved [{traj.shape[0]} frames x {traj.shape[1]} residues] → {args.output}')


if __name__ == '__main__':
    main()
