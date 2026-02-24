"""Autoregressive SE(3) video extrapolation.

Generates a trajectory of protein backbone frames by alternating between:
  - Conditional steps  (even): predict next frame conditioned on the previous frame
  - Refinement steps    (odd): remove artifacts from the conditional output via
                               SDEdit — forward-noise to t_start, then reverse
                               with the unconditional model

All SDE logic lives in inference_conditional.py and inference_unconditional.py.
This file only orchestrates the autoregressive loop and model loading.

Step pattern (0-indexed, starting from source frame):
    Frame 1 → conditional(source)
    Frame 2 → sdedit(frame 1)          ← artifact removal
    Frame 3 → conditional(frame 2)
    Frame 4 → sdedit(frame 3)
    ...

Output: ca_trajectory.npy — shape [num_frames + 1, N, 3] in Angstroms,
        centred at origin. Frame 0 is the source; frames 1..num_frames are
        generated.

Usage:
    python gen_model/video_extrapolation.py \\
        --conditional_ckpt   checkpoints/conditional_se3/best.ckpt \\
        --unconditional_ckpt checkpoints/4o66_C_se3/best.ckpt \\
        --npy_path           data/4o66_C/4o66_C_R1_latent.npy \\
        --atlas_csv          gen_model/splits/atlas.csv \\
        --protein_name       4o66_C \\
        --frame_idx          1000 \\
        --num_frames         20 \\
        --num_steps          100 \\
        --k                  1 \\
        --t_start            0.5 \\
        --out_dir            outputs/video_extrapolation
"""
import os
import argparse
import numpy as np
import torch

from gen_model.train_base import default_se3_conf, default_model_conf
from gen_model.train_unconditional import SE3Module
from gen_model.train_conditional import ConditionalSE3Module
from gen_model.diffusion.se3_diffuser import SE3Diffuser
from gen_model.inference_conditional import (
    run_reverse_sde as run_conditional_step,
    load_source_frame,
    compute_coord_scale,
)
from gen_model.inference_unconditional import (
    run_sdedit_step,
)


def build_trajectory(
    conditional_model,
    unconditional_model,
    diffuser,
    sc_ca_t: torch.Tensor,     # [N, 3] source CA positions (centred + scaled)
    aatype: torch.Tensor,       # [N]
    res_mask: torch.Tensor,     # [N]
    k: int,
    num_frames: int,
    num_steps: int,
    t_start: float,
    device: str,
    noise_scale: float = 1.0,
):
    """Generate num_frames frames autoregressively.

    Even steps (0, 2, ...): conditional — predict next frame from current sc_ca_t.
    Odd  steps (1, 3, ...): sdedit     — refine the previous conditional output by
                                         forward-noising to t_start and reverse-
                                         denoising with the unconditional model.

    The refined frame's translations are fed back as sc_ca_t for the next
    conditional step.

    Returns:
        List of Rigid objects (length num_frames), one per generated frame.
    """
    N = res_mask.shape[0]
    frames = []
    current_sc_ca_t    = sc_ca_t    # updated after every step
    last_cond_rigid    = None       # holds the most recent conditional output

    for i in range(num_frames):
        is_conditional = (i % 2 == 0)
        label = 'conditional' if is_conditional else f'sdedit(t_start={t_start})'
        print(f'\nFrame {i + 1}/{num_frames}  ({label})')

        if is_conditional:
            rigid = run_conditional_step(
                model=conditional_model,
                diffuser=diffuser,
                sc_ca_t=current_sc_ca_t,
                aatype=aatype,
                res_mask=res_mask,
                k=k,
                num_steps=num_steps,
                device=device,
                noise_scale=noise_scale,
            )
            last_cond_rigid = rigid
        else:
            # Refine the previous conditional output to remove artifacts
            rigid = run_sdedit_step(
                model=unconditional_model,
                diffuser=diffuser,
                starting_rigid=last_cond_rigid,
                aatype=aatype,
                res_mask=res_mask,
                N=N,
                t_start=t_start,
                num_steps=num_steps,
                device=device,
                noise_scale=noise_scale,
            )

        frames.append(rigid)

        # Feed generated frame's translations back as source for next conditional step.
        # get_trans() returns [N, 3] in centred+scaled space — same format as sc_ca_t.
        current_sc_ca_t = rigid.get_trans().float().detach().cpu()

    return frames


def main():
    parser = argparse.ArgumentParser(
        description='Autoregressive SE(3) video extrapolation (conditional + SDEdit refinement)'
    )
    parser.add_argument('--conditional_ckpt',   type=str, required=True,
                        help='Path to ConditionalSE3Module Lightning checkpoint')
    parser.add_argument('--unconditional_ckpt', type=str, required=True,
                        help='Path to SE3Module (unconditional) Lightning checkpoint')
    parser.add_argument('--npy_path',           type=str, required=True,
                        help='Trajectory .npy file for source frame + coord_scale')
    parser.add_argument('--atlas_csv',          type=str, default='gen_model/splits/atlas.csv')
    parser.add_argument('--protein_name',       type=str, required=True,
                        help='Protein name as used in atlas.csv (e.g. 4o66_C)')
    parser.add_argument('--frame_idx',          type=int, default=0,
                        help='Source frame index in the trajectory')
    parser.add_argument('--k',                  type=int, default=1,
                        help='Temporal gap for conditional steps')
    parser.add_argument('--num_frames',         type=int, default=20,
                        help='Number of frames to generate')
    parser.add_argument('--num_steps',          type=int, default=100,
                        help='Number of reverse SDE steps per frame')
    parser.add_argument('--t_start',            type=float, default=0.5,
                        help='Noise level for SDEdit refinement (0–1); '
                             'higher = more aggressive artifact removal')
    parser.add_argument('--coord_scale',        type=float, default=None,
                        help='Coordinate scale factor (default: computed from trajectory)')
    parser.add_argument('--lora_r',             type=int,   default=0)
    parser.add_argument('--lora_alpha',         type=float, default=0.0)
    parser.add_argument('--noise_scale',        type=float, default=1.0,
                        help='Noise multiplier in reverse SDE (0 = deterministic)')
    parser.add_argument('--out_dir',            type=str, default='outputs/video_extrapolation')
    parser.add_argument('--device',             type=str, default=None,
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

    # Source frame
    atom14, seqres, ca_pos, rigids_0, aatype = load_source_frame(
        args.npy_path, args.frame_idx, args.atlas_csv, args.protein_name
    )
    N        = len(seqres)
    centroid = ca_pos.mean(axis=0, keepdims=True)                              # [1, 3]
    sc_ca_t  = torch.from_numpy((ca_pos - centroid) * coord_scale).float()    # [N, 3]
    res_mask = torch.ones(N, dtype=torch.float32)
    print(f'Source frame : {args.protein_name}[{args.frame_idx}]  ({N} residues)')
    print(f't_start      : {args.t_start}  (SDEdit refinement noise level)')

    # Load models
    se3_conf    = default_se3_conf()
    cond_conf   = default_model_conf(use_temporal_embedding=True,
                                     lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    uncond_conf = default_model_conf(lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    conditional_model = ConditionalSE3Module.load_from_checkpoint(
        args.conditional_ckpt,
        model_conf=cond_conf, se3_conf=se3_conf, map_location=device,
    ).to(device).eval()

    unconditional_model = SE3Module.load_from_checkpoint(
        args.unconditional_ckpt,
        model_conf=uncond_conf, se3_conf=se3_conf, map_location=device,
    ).to(device).eval()

    diffuser = SE3Diffuser(se3_conf)

    print(f'Loaded conditional    : {args.conditional_ckpt}')
    print(f'Loaded unconditional  : {args.unconditional_ckpt}')

    # Generate trajectory
    frames = build_trajectory(
        conditional_model=conditional_model,
        unconditional_model=unconditional_model,
        diffuser=diffuser,
        sc_ca_t=sc_ca_t,
        aatype=aatype,
        res_mask=res_mask,
        k=args.k,
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        t_start=args.t_start,
        device=device,
        noise_scale=args.noise_scale,
    )

    # Convert to Angstroms (centred) and prepend source frame
    generated_ca = np.stack([
        f.get_trans().cpu().numpy() / coord_scale
        for f in frames
    ], axis=0)                                           # [num_frames, N, 3]

    source_ca  = (ca_pos - centroid).astype(np.float32) # [N, 3] centred Angstroms
    trajectory = np.concatenate(
        [source_ca[None], generated_ca], axis=0
    )                                                    # [num_frames + 1, N, 3]

    out_path = os.path.join(args.out_dir, 'ca_trajectory.npy')
    np.save(out_path, trajectory)

    pattern = ' '.join('C' if i % 2 == 0 else 'R' for i in range(args.num_frames))
    print(f'\n✓  Saved trajectory : {out_path}  shape={trajectory.shape}')
    print(f'   Frame 0 = source, Frames 1–{args.num_frames} = generated')
    print(f'   Step pattern      : {pattern}  (C=conditional, R=refinement)')


if __name__ == '__main__':
    main()
