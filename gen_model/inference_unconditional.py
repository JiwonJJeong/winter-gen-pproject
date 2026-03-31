"""Unconditional SE(3) inference: generate protein backbone conformations from noise.

Loads a trained SE3Diffusion (unconditional) checkpoint and runs the reverse SE(3) SDE
from pure noise, producing new backbone conformations sampled from p(x).

The output is a CA-trace saved as a .npy array [num_samples, N, 3] in Angstroms,
centred at the origin.

Usage:
    python gen_model/inference_unconditional.py \\
        --checkpoint checkpoints/4o66_C_se3/best.ckpt \\
        --npy_path data/4o66_C/4o66_C_R1_latent.npy \\
        --atlas_csv gen_model/splits/atlas.csv \\
        --protein_name 4o66_C \\
        --num_steps 100 \\
        --num_samples 5 \\
        --out_dir outputs/inference_unconditional
"""
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from gen_model.train_base import default_se3_conf, default_model_conf
from gen_model.se3_diffusion_module import SE3Diffusion
from gen_model.diffusion.se3_diffuser import SE3Diffuser, _assemble_rigid
from gen_model.data.residue_constants import restype_order
from gen_model.inference_conditional import compute_coord_scale


@torch.no_grad()
def run_reverse_sde(
    model,
    diffuser: SE3Diffuser,
    aatype: torch.Tensor,       # [N]
    res_mask: torch.Tensor,     # [N]
    N: int,
    num_steps: int,
    device: str,
    noise_scale: float = 1.0,
):
    """Run the unconditional SE(3) reverse SDE for `num_steps` steps.

    Starts from the reference distribution (t=1) and iterates to t=0.
    sc_ca_t is all zeros (no source frame conditioning).
    Returns the predicted Rigid frames at t=0.
    """
    model.eval()

    # Sample initial noise at t=1
    rot_ref   = diffuser._so3_diffuser.sample_ref(n_samples=N)
    trans_ref = diffuser._r3_diffuser.sample_ref(n_samples=N)
    rigid_t   = _assemble_rigid(rot_ref, trans_ref)

    ts = np.linspace(1.0, 0.0, num_steps + 1)
    dt = float(ts[0] - ts[1])

    # Batch constants (batch size = 1)
    sc_ca_t_b   = torch.zeros(1, N, 3, device=device)              # unconditional
    res_mask_b  = res_mask.unsqueeze(0).to(device)
    fixed_mask  = torch.zeros_like(res_mask_b)
    aatype_b    = aatype.unsqueeze(0).to(device)
    seq_idx_b   = torch.arange(1, N + 1, device=device).unsqueeze(0)
    chain_idx_b = torch.zeros(1, N, dtype=torch.long, device=device)

    for t_val in tqdm(ts[:-1], desc='Reverse SDE'):
        t_tensor    = torch.tensor([t_val], dtype=torch.float32, device=device)
        rigids_t_b  = torch.from_numpy(
            rigid_t.to_tensor_7().numpy()
        ).unsqueeze(0).to(device)                                   # [1, N, 7]

        input_feats = {
            'rigids_t':               rigids_t_b,
            'sc_ca_t':                sc_ca_t_b,
            'res_mask':               res_mask_b,
            'fixed_mask':             fixed_mask,
            't':                      t_tensor,
            'aatype':                 aatype_b,
            'seq_idx':                seq_idx_b,
            'chain_idx':              chain_idx_b,
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

    return rigid_t


@torch.no_grad()
def run_sdedit_step(
    model,
    diffuser: SE3Diffuser,
    starting_rigid,             # Rigid at t=0 (conditional output to refine)
    aatype: torch.Tensor,       # [N]
    res_mask: torch.Tensor,     # [N]
    N: int,
    t_start: float,             # noise level to apply (0.0–1.0); higher = more aggressive
    num_steps: int,
    device: str,
    noise_scale: float = 1.0,
):
    """SDEdit-style artifact removal using the unconditional model.

    1. Forward-noises starting_rigid to t_start using the SE(3) forward process.
    2. Reverse-denoises from t_start back to t=0 with the unconditional model.

    Higher t_start removes more artifacts but drifts further from the input.
    Lower t_start is a conservative correction that preserves more of the input.
    """
    model.eval()

    # Forward: noise the conditional output to t_start
    noised  = diffuser.forward_marginal(starting_rigid, t_start, as_tensor_7=False)
    rigid_t = noised['rigids_t']   # Rigid at t_start

    # Reverse: denoise from t_start to 0
    ts = np.linspace(t_start, 0.0, num_steps + 1)
    dt = float(ts[0] - ts[1])

    # Batch constants (batch size = 1)
    sc_ca_t_b   = torch.zeros(1, N, 3, device=device)
    res_mask_b  = res_mask.unsqueeze(0).to(device)
    fixed_mask  = torch.zeros_like(res_mask_b)
    aatype_b    = aatype.unsqueeze(0).to(device)
    seq_idx_b   = torch.arange(1, N + 1, device=device).unsqueeze(0)
    chain_idx_b = torch.zeros(1, N, dtype=torch.long, device=device)

    for t_val in tqdm(ts[:-1], desc='Refinement SDE'):
        t_tensor   = torch.tensor([t_val], dtype=torch.float32, device=device)
        rigids_t_b = torch.from_numpy(
            rigid_t.to_tensor_7().numpy()
        ).unsqueeze(0).to(device)                                    # [1, N, 7]

        input_feats = {
            'rigids_t':               rigids_t_b,
            'sc_ca_t':                sc_ca_t_b,
            'res_mask':               res_mask_b,
            'fixed_mask':             fixed_mask,
            't':                      t_tensor,
            'aatype':                 aatype_b,
            'seq_idx':                seq_idx_b,
            'chain_idx':              chain_idx_b,
            # gt_psi is multiplied by 0 during inference (fixed_mask=0 → diff_mask=1)
            # so zeros are fine; the key must exist to avoid a KeyError in score_network.
            'torsion_angles_sin_cos': torch.zeros(1, N, 7, 2, device=device),
        }

        pred = model.model(input_feats)

        rot_score   = pred['rot_score'][0].cpu().numpy()
        trans_score = pred['trans_score'][0].cpu().numpy()

        rigid_t = diffuser.reverse(
            rigid_t=rigid_t,
            rot_score=rot_score,
            trans_score=trans_score,
            t=t_val,
            dt=dt,
            center=True,
            noise_scale=noise_scale,
        )

    return rigid_t


def main():
    parser = argparse.ArgumentParser(description='Unconditional SE(3) inference — sample from noise')
    parser.add_argument('--checkpoint',   type=str, required=True,
                        help='Path to SE3Diffusion Lightning checkpoint')
    parser.add_argument('--npy_path',     type=str, required=True,
                        help='Path to trajectory .npy file (used to compute coord_scale)')
    parser.add_argument('--atlas_csv',    type=str, default='gen_model/splits/atlas.csv')
    parser.add_argument('--protein_name', type=str, required=True,
                        help='Protein name as used in atlas.csv (e.g. 4o66_C)')
    parser.add_argument('--num_steps',    type=int, default=100,
                        help='Number of reverse SDE steps')
    parser.add_argument('--num_samples',  type=int, default=100,
                        help='Number of independent samples to generate (default: 100 for evaluation)')
    parser.add_argument('--coord_scale',  type=float, default=None,
                        help='Coordinate scale factor (default: computed from trajectory)')
    parser.add_argument('--lora_r',       type=int,   default=0,
                        help='LoRA rank used during training (0 = no LoRA)')
    parser.add_argument('--lora_alpha',   type=float, default=0.0)
    parser.add_argument('--noise_scale',  type=float, default=1.0,
                        help='Noise multiplier in reverse SDE (0 = deterministic)')
    parser.add_argument('--out_dir',      type=str, default='outputs/inference_unconditional')
    parser.add_argument('--device',       type=str, default=None,
                        help='cuda / cpu (default: auto-detect)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # coord_scale
    if args.coord_scale is not None:
        coord_scale = args.coord_scale
        print(f'coord_scale : {coord_scale:.4f}  (from --coord_scale)')
    else:
        coord_scale = compute_coord_scale(args.npy_path)
        print(f'coord_scale : {coord_scale:.4f}  (computed from trajectory)')

    # aatype and N from atlas_csv seqres
    import pandas as pd
    seq_df  = pd.read_csv(args.atlas_csv, index_col='name')
    seqres  = seq_df.loc[args.protein_name, 'seqres']
    N       = len(seqres)
    aatype  = torch.tensor([restype_order[c] for c in seqres], dtype=torch.long)
    res_mask = torch.ones(N, dtype=torch.float32)
    print(f'Protein : {args.protein_name}  ({N} residues)')

    # Load model
    se3_conf   = default_se3_conf()
    model_conf = default_model_conf(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    diffuser   = SE3Diffuser(se3_conf)

    from gen_model.models.star_score_network import StarScoreNetwork
    from gen_model.models.lora import apply_lora
    score_network = StarScoreNetwork(model_conf, diffuser)
    apply_lora(score_network, model_conf.lora)

    module = SE3Diffusion(
        model=score_network,
        diffuser=diffuser,
    )

    # Load checkpoint weights
    ckpt  = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    module.load_state_dict(state, strict=False)
    module = module.to(device)
    module.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    # Generate samples
    all_ca = []
    for s in range(args.num_samples):
        print(f'\nSample {s + 1}/{args.num_samples}')
        rigid_pred = run_reverse_sde(
            model=module,
            diffuser=diffuser,
            aatype=aatype,
            res_mask=res_mask,
            N=N,
            num_steps=args.num_steps,
            device=device,
            noise_scale=args.noise_scale,
        )
        # Unscale translations; output is centred at origin
        trans_pred       = rigid_pred.get_trans().cpu().numpy()  # [N, 3] scaled
        ca_pred_angstrom = trans_pred / coord_scale               # [N, 3] Angstroms
        all_ca.append(ca_pred_angstrom)

    all_ca  = np.stack(all_ca, axis=0)   # [num_samples, N, 3]
    ca_path = os.path.join(args.out_dir, 'ca_samples.npy')
    np.save(ca_path, all_ca)
    print(f'\n✓  Generated CA positions: {ca_path}  shape={all_ca.shape}')
    print(f'   (coordinates are centred at origin; translate as needed)')


if __name__ == '__main__':
    main()
