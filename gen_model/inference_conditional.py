"""Conditional SE(3) inference: generate the next frame from a source frame.

Loads a trained ConditionalSE3Module checkpoint and runs the reverse SE(3) SDE
conditioned on a source backbone frame, producing a prediction of the frame
k MD strides ahead (or behind if k < 0).

The output is a CA-trace trajectory saved as a .npy array [num_samples, N, 3]
in physical Angstrom coordinates.

Usage:
    python gen_model/inference_conditional.py \\
        --checkpoint checkpoints/conditional_se3/best.ckpt \\
        --npy_path data/4o66_C/4o66_C_R1_latent.npy \\
        --atlas_csv gen_model/splits/atlas.csv \\
        --protein_name 4o66_C \\
        --frame_idx 1000 \\
        --k 1 \\
        --num_steps 100 \\
        --num_samples 5 \\
        --out_dir outputs/inference
"""
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from gen_model.train_base import default_se3_conf, default_model_conf
from gen_model.train_conditional import ConditionalSE3Module
from gen_model.diffusion.se3_diffuser import SE3Diffuser, _assemble_rigid, _extract_trans_rots
from gen_model.data.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from gen_model.data.residue_constants import restype_order
from gen_model.utils.rigid_utils import Rigid, Rotation


# ---------------------------------------------------------------------------
# Source frame loading
# ---------------------------------------------------------------------------

def load_source_frame(npy_path: str, frame_idx: int, atlas_csv: str, protein_name: str):
    """Load a single frame from a trajectory .npy file and compute its geometry.

    Returns:
        atom14: [L, 14, 3] float32 in Angstroms
        seqres: str amino-acid sequence
        ca_pos: [L, 3] Cα positions in Angstroms
        rigids_0: Rigid object with backbone frames [L]
        aatype: [L] int64 residue type indices
    """
    import pandas as pd

    seq_df = pd.read_csv(atlas_csv, index_col='name')
    folder = protein_name  # e.g. '4o66_C'
    seqres = seq_df.loc[folder, 'seqres']
    L = len(seqres)

    arr = np.lib.format.open_memmap(npy_path, 'r')
    atom14 = np.copy(arr[frame_idx]).astype(np.float32)  # [L, 14, 3]

    aatype_np = np.array([restype_order[c] for c in seqres], dtype=np.int64)
    aatype = torch.from_numpy(aatype_np)

    atom14_t = torch.from_numpy(atom14)[None]          # [1, L, 14, 3]
    aatype_t = torch.from_numpy(aatype_np)[None]       # [1, L]
    frames   = atom14_to_frames(atom14_t)
    rigids_0 = Rigid(
        rots=Rotation(rot_mats=frames._rots._rot_mats[0]),
        trans=frames._trans[0],
    )
    ca_pos = atom14[..., 1, :]  # Cα is atom index 1 in atom14

    return atom14, seqres, ca_pos, rigids_0, aatype


def compute_coord_scale(npy_path: str, max_frames: int = 500) -> float:
    """Estimate coord_scale = 1/std(centred Cα) from the trajectory.

    Mirrors the logic in MDGenDataset._compute_coord_scale so that inference
    uses the same normalisation as training (assuming a single-protein setup
    where training and inference draw from the same trajectory).
    """
    arr   = np.lib.format.open_memmap(npy_path, 'r')
    n     = min(max_frames, len(arr))
    idxs  = np.random.choice(len(arr), n, replace=False)
    cas   = []
    for i in idxs:
        ca = arr[i][:, 1, :].astype(np.float32)
        ca = ca - ca.mean(axis=0, keepdims=True)
        cas.append(ca)
    all_ca = np.concatenate(cas, axis=0)
    std = np.std(all_ca)
    return float(1.0 / (std + 1e-8))


# ---------------------------------------------------------------------------
# Reverse SDE
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_reverse_sde(
    model,
    diffuser: SE3Diffuser,
    sc_ca_t: torch.Tensor,          # [N, 3] source CA positions (centred + scaled)
    aatype: torch.Tensor,            # [N]
    res_mask: torch.Tensor,          # [N]
    k: int,
    num_steps: int,
    device: str,
    noise_scale: float = 1.0,
) -> Rigid:
    """Run the SE(3) reverse SDE for `num_steps` steps.

    Starts from the reference distribution (t=1 noise) and iterates to t=0.
    Returns the predicted rigid frames at t=0.
    """
    N = sc_ca_t.shape[0]
    model.eval()

    # ── Sample initial noise at t=1 ──────────────────────────────────────────
    rot_ref  = diffuser._so3_diffuser.sample_ref(n_samples=N)    # [N, 3] rotvec
    trans_ref = diffuser._r3_diffuser.sample_ref(n_samples=N)    # [N, 3] ~ N(0,I)
    rigid_t  = _assemble_rigid(rot_ref, trans_ref)               # Rigid [N]

    # ── Time schedule ─────────────────────────────────────────────────────────
    ts = np.linspace(1.0, 0.0, num_steps + 1)
    dt = float(ts[0] - ts[1])

    # ── Batch constants (add batch dim) ───────────────────────────────────────
    sc_ca_t_b   = sc_ca_t.unsqueeze(0).to(device)       # [1, N, 3]
    res_mask_b  = res_mask.unsqueeze(0).to(device)       # [1, N]
    fixed_mask  = torch.zeros_like(res_mask_b)           # [1, N] — all diffused
    aatype_b    = aatype.unsqueeze(0).to(device)         # [1, N]
    seq_idx_b   = torch.arange(N, device=device).unsqueeze(0)   # [1, N]
    chain_idx_b = torch.zeros(1, N, dtype=torch.long, device=device)
    k_b         = torch.tensor([k], dtype=torch.long, device=device)

    for i, t_val in enumerate(tqdm(ts[:-1], desc='Reverse SDE')):
        t_tensor = torch.tensor([t_val], dtype=torch.float32, device=device)

        # Build rigids_t as tensor_7 with batch dim
        rigids_t_np = rigid_t.to_tensor_7().numpy()           # [N, 7]
        rigids_t_b  = torch.from_numpy(rigids_t_np).unsqueeze(0).to(device)  # [1, N, 7]

        input_feats = {
            'rigids_t':               rigids_t_b,
            'sc_ca_t':                sc_ca_t_b,
            'res_mask':               res_mask_b,
            'fixed_mask':             fixed_mask,
            't':                      t_tensor,
            'aatype':                 aatype_b,
            'seq_idx':                seq_idx_b,
            'chain_idx':              chain_idx_b,
            'k':                      k_b,
            # gt_psi is multiplied by 0 during inference (fixed_mask=0 → diff_mask=1)
            # so zeros are fine; the key must exist to avoid a KeyError in score_network.
            'torsion_angles_sin_cos': torch.zeros(1, N, 7, 2, device=device),
        }

        pred = model.model(input_feats)

        rot_score   = pred['rot_score'][0].cpu().numpy()    # [N, 3]
        trans_score = pred['trans_score'][0].cpu().numpy()  # [N, 3]

        rigid_t = diffuser.reverse(
            rigid_t=rigid_t,
            rot_score=rot_score,
            trans_score=trans_score,
            t=t_val,
            dt=dt,
            center=True,
            noise_scale=noise_scale,
        )

    return rigid_t   # Rigid at t=0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Conditional SE(3) inference — next frame prediction')
    parser.add_argument('--checkpoint',   type=str, required=True,
                        help='Path to ConditionalSE3Module Lightning checkpoint')
    parser.add_argument('--npy_path',     type=str, required=True,
                        help='Path to trajectory .npy file (source + reference for coord_scale)')
    parser.add_argument('--atlas_csv',    type=str, default='gen_model/splits/atlas.csv')
    parser.add_argument('--protein_name', type=str, required=True,
                        help='Protein name as used in atlas.csv (e.g. 4o66_C)')
    parser.add_argument('--frame_idx',    type=int, default=0,
                        help='Source frame index in the trajectory')
    parser.add_argument('--k',            type=int, default=1,
                        help='Temporal gap: 1 = next frame, -1 = previous, etc.')
    parser.add_argument('--num_steps',    type=int, default=100,
                        help='Number of reverse SDE steps')
    parser.add_argument('--num_samples',  type=int, default=5,
                        help='Number of independent samples to generate')
    parser.add_argument('--coord_scale',  type=float, default=None,
                        help='Coordinate scale factor (default: computed from trajectory)')
    parser.add_argument('--lora_r',       type=int,   default=0,
                        help='LoRA rank used during training (0 = no LoRA)')
    parser.add_argument('--lora_alpha',   type=float, default=0.0)
    parser.add_argument('--noise_scale',  type=float, default=1.0,
                        help='Noise multiplier in reverse SDE (0 = deterministic DDIM-like)')
    parser.add_argument('--out_dir',      type=str, default='outputs/inference_conditional')
    parser.add_argument('--device',       type=str, default=None,
                        help='cuda / cpu (default: auto-detect)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── coord_scale ───────────────────────────────────────────────────────────
    if args.coord_scale is not None:
        coord_scale = args.coord_scale
        print(f'coord_scale : {coord_scale:.4f}  (from --coord_scale)')
    else:
        coord_scale = compute_coord_scale(args.npy_path)
        print(f'coord_scale : {coord_scale:.4f}  (computed from trajectory)')

    # ── Load source frame ─────────────────────────────────────────────────────
    atom14, seqres, ca_pos, rigids_0, aatype = load_source_frame(
        args.npy_path, args.frame_idx, args.atlas_csv, args.protein_name
    )
    N = len(seqres)
    print(f'Source frame: {args.npy_path}[{args.frame_idx}]  ({N} residues, k={args.k})')

    # Centre and scale source CA positions
    centroid   = ca_pos.mean(axis=0, keepdims=True)            # [1, 3]
    sc_ca_t    = torch.from_numpy((ca_pos - centroid) * coord_scale).float()  # [N, 3]
    res_mask   = torch.ones(N, dtype=torch.float32)

    # ── Load model ────────────────────────────────────────────────────────────
    se3_conf   = default_se3_conf()
    model_conf = default_model_conf(
        use_temporal_embedding=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    module = ConditionalSE3Module.load_from_checkpoint(
        args.checkpoint,
        model_conf=model_conf,
        se3_conf=se3_conf,
        map_location=device,
    )
    module = module.to(device)
    module.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    diffuser = SE3Diffuser(se3_conf)

    # ── Generate samples ──────────────────────────────────────────────────────
    all_ca = []  # collect [num_samples, N, 3] in Angstroms

    for s in range(args.num_samples):
        print(f'\nSample {s + 1}/{args.num_samples}')
        rigid_pred = run_reverse_sde(
            model=module,
            diffuser=diffuser,
            sc_ca_t=sc_ca_t,
            aatype=aatype,
            res_mask=res_mask,
            k=args.k,
            num_steps=args.num_steps,
            device=device,
            noise_scale=args.noise_scale,
        )

        # Unscale translations → CA positions in Angstroms
        trans_pred = rigid_pred.get_trans().cpu().numpy()     # [N, 3] in coord_scale space
        ca_pred_angstrom = trans_pred / coord_scale + centroid  # [N, 3] in Angstroms
        all_ca.append(ca_pred_angstrom)

    all_ca = np.stack(all_ca, axis=0)  # [num_samples, N, 3]

    # ── Save outputs ──────────────────────────────────────────────────────────
    tag = f'frame{args.frame_idx}_k{args.k}'
    ca_path     = os.path.join(args.out_dir, f'{tag}_ca_pred.npy')
    source_path = os.path.join(args.out_dir, f'{tag}_ca_source.npy')

    np.save(ca_path,     all_ca)
    np.save(source_path, ca_pos)

    print(f'\n✓  Generated CA positions : {ca_path}  shape={all_ca.shape}')
    print(f'✓  Source CA positions    : {source_path}  shape={ca_pos.shape}')

    # Quick RMSD vs source to sanity-check scale
    rmsd = np.sqrt(np.mean((all_ca - ca_pos[None]) ** 2))
    print(f'\nMean RMSD to source frame : {rmsd:.2f} Å  '
          f'(expected ~several Å for k={args.k})')


if __name__ == '__main__':
    main()
