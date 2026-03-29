"""Evaluation suite for generated MD trajectories.

Converts generated trajectories to PDB/XTC format, then delegates metric
computation to MDGen's analyze_peptide_sim.py (extern/mdgen/scripts/).

Metrics computed by MDGen:
  - Torsion JSD (backbone + sidechain, 1D and 2D phi-psi)
  - TICA JSD (1D and 2D)
  - Torsion autocorrelation / decorrelation
  - TICA autocorrelation

For unconditional (i.i.d. samples), autocorrelation is skipped via --no_decorr.

Following ATLAS evaluation protocol: 250-frame trajectories for conditional,
100 i.i.d. samples for unconditional.

Usage:
    # Conditional (250-frame trajectory)
    python gen_model/evaluate.py \\
        --ref_npy data/4o66_C/4o66_C_R1_latent.npy \\
        --gen_traj outputs/conditional/4o66_C/traj.pt \\
        --atlas_csv gen_model/splits/atlas.csv \\
        --protein 4o66_C \\
        --mode conditional

    # Unconditional (100 i.i.d. samples)
    python gen_model/evaluate.py \\
        --ref_npy data/4o66_C/4o66_C_R1_latent.npy \\
        --gen_traj outputs/unconditional/4o66_C/ca_samples.npy \\
        --atlas_csv gen_model/splits/atlas.csv \\
        --protein 4o66_C \\
        --mode unconditional
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

import gen_model.path_setup  # noqa: F401

from mdgen.utils import atom14_to_pdb
from mdgen.residue_constants import restype_order


# ---------------------------------------------------------------------------
# Trajectory conversion: our formats → PDB + XTC for pyemma/MDGen
# ---------------------------------------------------------------------------

def _load_ref_atom14(npy_path: str, split_csv: str = None,
                     mode: str = 'val') -> np.ndarray:
    """Load reference trajectory as atom14 array [T, N, 14, 3].

    If split_csv is provided, only the val/test segment is used.
    """
    arr = np.load(npy_path).astype(np.float32)
    if split_csv is not None:
        name = os.path.basename(npy_path).replace('_latent.npy', '')
        df = pd.read_csv(split_csv, index_col='name')
        if name in df.index and 'train_end' in df.columns:
            t_e = int(df.loc[name, 'train_end'])
            v_e = int(df.loc[name, 'val_end'])
            if mode == 'val':
                arr = arr[t_e:v_e]
            elif mode == 'test':
                arr = arr[v_e:]
    return arr


def _rigids_to_atom14(gen_path: str, ref_npy_path: str,
                      atlas_csv: str, protein: str) -> tuple:
    """Convert generated .pt ([T, N, 7] rigids) to atom14 [T, N, 14, 3].

    Extracts CA from the translation component and approximates other
    backbone atoms by shifting from the reference first frame.
    """
    traj = torch.load(gen_path, map_location='cpu')  # [T, N, 7]
    T, N, _ = traj.shape

    seq_df = pd.read_csv(atlas_csv, index_col='name')
    seqres = seq_df.loc[protein, 'seqres']
    aatype = np.array([restype_order[c] for c in seqres])

    # Recover coord_scale from reference
    ref_arr = np.load(ref_npy_path).astype(np.float32)
    ref_ca = ref_arr[:, :, 1, :]
    ref_ca_centered = ref_ca - ref_ca.mean(axis=1, keepdims=True)
    coord_scale = 1.0 / (np.std(ref_ca_centered) + 1e-8)

    # Unscale translations back to Angstroms
    ca_gen = traj[:, :, 4:7].numpy() / coord_scale

    # Build atom14: CA at index 1, approximate other backbone from ref
    atom14 = np.zeros((T, N, 14, 3), dtype=np.float32)
    atom14[:, :, 1, :] = ca_gen
    ref_frame = ref_arr[0]
    ref_ca_0 = ref_frame[:, 1, :]
    for t in range(T):
        offset = ca_gen[t] - ref_ca_0
        for atom_idx in [0, 2, 3, 4]:  # N, C, O, CB
            atom14[t, :, atom_idx, :] = ref_frame[:, atom_idx, :] + offset

    return atom14, aatype


def _ca_samples_to_atom14(gen_path: str, ref_npy_path: str,
                          atlas_csv: str, protein: str) -> tuple:
    """Convert unconditional CA samples .npy ([S, N, 3]) to atom14.

    Same backbone approximation as _rigids_to_atom14.
    """
    ca_samples = np.load(gen_path).astype(np.float32)  # [S, N, 3]
    S, N, _ = ca_samples.shape

    seq_df = pd.read_csv(atlas_csv, index_col='name')
    seqres = seq_df.loc[protein, 'seqres']
    aatype = np.array([restype_order[c] for c in seqres])

    ref_arr = np.load(ref_npy_path).astype(np.float32)
    ref_frame = ref_arr[0]
    ref_ca_0 = ref_frame[:, 1, :]

    atom14 = np.zeros((S, N, 14, 3), dtype=np.float32)
    atom14[:, :, 1, :] = ca_samples
    for s in range(S):
        offset = ca_samples[s] - ref_ca_0
        for atom_idx in [0, 2, 3, 4]:
            atom14[s, :, atom_idx, :] = ref_frame[:, atom_idx, :] + offset

    return atom14, aatype


def _write_trajectory(atom14: np.ndarray, aatype: np.ndarray,
                      out_dir: str, name: str):
    """Write atom14 trajectory as PDB + XTC files for pyemma."""
    import mdtraj

    pdb_path = os.path.join(out_dir, f'{name}.pdb')
    xtc_path = os.path.join(out_dir, f'{name}.xtc')

    atom14_to_pdb(atom14, aatype, pdb_path)

    traj = mdtraj.load(pdb_path)
    traj.superpose(traj)
    traj[0].save(pdb_path)
    traj.save(xtc_path)

    return pdb_path, xtc_path


# ---------------------------------------------------------------------------
# Structural validity (STAR-MD paper appendix)
# ---------------------------------------------------------------------------

def compute_validity(atom14: np.ndarray, aatype: np.ndarray) -> dict:
    """Compute structural validity metrics following STAR-MD paper.

    Args:
        atom14: [T, N, 14, 3] generated atom14 coordinates
        aatype: [N] residue type indices

    Returns:
        dict with ca_clash_rate, chain_break_rate, ca_valid_rate, per-frame details
    """
    CA_CLASH_THRESHOLD = 2.0    # Angstroms (paper)
    CHAIN_BREAK_THRESHOLD = 3.8  # Angstroms (paper)
    T, N = atom14.shape[:2]

    ca = atom14[:, :, 1, :]  # [T, N, 3] — atom14 index 1 = CA

    ca_clashes = 0
    chain_breaks = 0
    total_pairs = 0
    total_bonds = 0

    for t in range(T):
        ca_t = ca[t]  # [N, 3]

        # Ca-Ca clashes: any pair of non-adjacent CA atoms closer than 2.0 Å
        for i in range(N):
            for j in range(i + 2, N):  # skip adjacent (i, i+1)
                d = np.linalg.norm(ca_t[i] - ca_t[j])
                if d < CA_CLASH_THRESHOLD and d > 0.1:  # skip zero-distance (padding)
                    ca_clashes += 1
                total_pairs += 1

        # Chain breaks: consecutive CA atoms farther than 3.8 Å
        for i in range(N - 1):
            d = np.linalg.norm(ca_t[i + 1] - ca_t[i])
            if d > CHAIN_BREAK_THRESHOLD or d < 0.1:  # too far or zero (padding)
                chain_breaks += 1
            total_bonds += 1

    ca_clash_rate = ca_clashes / max(total_pairs, 1)
    chain_break_rate = chain_breaks / max(total_bonds, 1)
    ca_valid_rate = 1.0 - ca_clash_rate
    aa_valid_rate = 1.0 - chain_break_rate
    combined_valid = ca_valid_rate * aa_valid_rate * 100

    return {
        'ca_clash_rate': ca_clash_rate,
        'chain_break_rate': chain_break_rate,
        'ca_valid_pct': ca_valid_rate * 100,
        'aa_valid_pct': aa_valid_rate * 100,
        'combined_valid_pct': combined_valid,
        'num_frames': T,
        'num_residues': N,
    }


# ---------------------------------------------------------------------------
# Run MDGen analysis
# ---------------------------------------------------------------------------

def run_mdgen_analysis(ref_dir: str, gen_dir: str, protein: str,
                       mode: str = 'conditional',
                       out_dir: str = 'outputs/eval',
                       plot: bool = True) -> str:
    """Call MDGen's analyze_peptide_sim.py on the prepared PDB/XTC files.

    Args:
        ref_dir:  Directory with {protein}.pdb/.xtc (reference)
        gen_dir:  Directory with {protein}.pdb/.xtc (generated)
        protein:  Protein name
        mode:     'conditional' or 'unconditional'
        out_dir:  Where to save results
        plot:     Generate plots

    Returns:
        Path to the saved .pkl results file.
    """
    script = os.path.join(
        os.path.dirname(__file__), '..', 'extern', 'mdgen', 'scripts',
        'analyze_peptide_sim.py')
    script = os.path.abspath(script)

    save_name = 'eval_results.pkl'

    cmd = [
        sys.executable, script,
        '--mddir', ref_dir,
        '--pdbdir', gen_dir,
        '--pdb_id', protein,
        '--save',
        '--save_name', save_name,
        '--no_msm',           # skip MSM/TPS (not needed)
    ]

    if mode == 'unconditional':
        cmd.append('--no_decorr')  # i.i.d. samples have no temporal structure

    if plot:
        cmd.append('--plot')

    print(f'Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f'MDGen analysis failed (exit {result.returncode}).\n'
            f'stderr: {result.stderr}')

    pkl_path = os.path.join(gen_dir, save_name)
    return pkl_path


def print_summary(pkl_path: str, mode: str, validity: dict = None):
    """Load and print a human-readable summary from the MDGen results pickle."""
    import pickle

    with open(pkl_path, 'rb') as f:
        all_results = pickle.load(f)

    print('\n' + '=' * 60)
    print(f'EVALUATION SUMMARY  ({mode} mode)')
    print('=' * 60)

    if validity:
        print(f'\nStructural Validity (STAR-MD paper metrics):')
        print(f'  CA valid (no clashes <2.0A):     {validity["ca_valid_pct"]:.1f}%')
        print(f'  AA valid (no chain breaks >3.8A): {validity["aa_valid_pct"]:.1f}%')
        print(f'  Combined (CA * AA):               {validity["combined_valid_pct"]:.1f}%')

    for protein, results in all_results.items():
        print(f'\nProtein: {protein}')

        if 'JSD' not in results:
            print('  No JSD results found.')
            continue

        jsd = results['JSD']

        backbone = {k: v for k, v in jsd.items()
                    if 'PHI' in k or 'PSI' in k or 'OMEGA' in k or '|' in k}
        sidechain = {k: v for k, v in jsd.items() if 'CHI' in k}
        tica = {k: v for k, v in jsd.items() if k.startswith('TICA')}

        if backbone:
            bb_mean = np.mean(list(backbone.values()))
            print(f'\n  Backbone torsion JSD (mean): {bb_mean:.4f}')
            for k, v in sorted(backbone.items()):
                print(f'    {k:40s}  {v:.4f}')

        if sidechain:
            sc_mean = np.mean(list(sidechain.values()))
            print(f'\n  Sidechain torsion JSD (mean): {sc_mean:.4f}')
            for k, v in sorted(sidechain.items()):
                print(f'    {k:40s}  {v:.4f}')

        if tica:
            print(f'\n  TICA JSD:')
            for k, v in sorted(tica.items()):
                print(f'    {k:40s}  {v:.4f}')

        if mode == 'conditional' and 'our_decorrelation' in results:
            print(f'\n  Autocorrelation computed (see {pkl_path} for full data)')

    print('=' * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate generated MD trajectory against reference')
    parser.add_argument('--ref_npy',    type=str, required=True, nargs='+',
                        help='Reference trajectory .npy file(s). Pass multiple replicas '
                             'to evaluate against the combined ensemble, e.g.: '
                             '--ref_npy data/4o66_C/4o66_C_R{1,2,3}_latent.npy')
    parser.add_argument('--gen_traj',   type=str, required=True,
                        help='Generated trajectory: .pt (conditional) or .npy (unconditional)')
    parser.add_argument('--atlas_csv',  type=str, default='gen_model/splits/atlas.csv')
    parser.add_argument('--protein',    type=str, required=True)
    parser.add_argument('--mode',       type=str, default='conditional',
                        choices=['conditional', 'unconditional'],
                        help='conditional: trajectory with temporal structure; '
                             'unconditional: i.i.d. samples (skips autocorrelation)')
    parser.add_argument('--split_csv',  type=str, default=None,
                        help='Frame splits CSV (to use only val/test segment of ref)')
    parser.add_argument('--ref_mode',   type=str, default='val',
                        choices=['val', 'test', 'all'],
                        help='Which segment of reference to use')
    parser.add_argument('--no_plot',    action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--out_dir',    type=str, default='outputs/eval')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load reference (concatenate all replicas) ---
    ref_parts = []
    for npy_path in args.ref_npy:
        print(f'Loading reference: {npy_path}')
        if args.ref_mode == 'all':
            part = np.load(npy_path).astype(np.float32)
        else:
            part = _load_ref_atom14(
                npy_path, split_csv=args.split_csv, mode=args.ref_mode)
        ref_parts.append(part)
        print(f'  {os.path.basename(npy_path)}: {part.shape[0]} frames')
    ref_atom14 = np.concatenate(ref_parts, axis=0)
    print(f'  Total reference frames: {ref_atom14.shape[0]} '
          f'({len(args.ref_npy)} replica{"s" if len(args.ref_npy) > 1 else ""})')

    # Get aatype
    seq_df = pd.read_csv(args.atlas_csv, index_col='name')
    seqres = seq_df.loc[args.protein, 'seqres']
    aatype = np.array([restype_order[c] for c in seqres])

    # Use first replica for coord_scale reference
    first_ref_npy = args.ref_npy[0]

    # --- Load and convert generated ---
    print(f'Loading generated: {args.gen_traj}')
    if args.mode == 'conditional':
        gen_atom14, _ = _rigids_to_atom14(
            args.gen_traj, first_ref_npy, args.atlas_csv, args.protein)
    else:
        gen_atom14, _ = _ca_samples_to_atom14(
            args.gen_traj, first_ref_npy, args.atlas_csv, args.protein)
    print(f'  Generated frames: {gen_atom14.shape[0]}')

    # --- Write PDB/XTC ---
    # MDGen expects: --mddir has {protein}/{protein}.pdb/.xtc
    #                --pdbdir has {protein}.pdb/.xtc
    ref_traj_dir = os.path.join(args.out_dir, 'ref', args.protein)
    gen_traj_dir = os.path.join(args.out_dir, 'gen')
    os.makedirs(ref_traj_dir, exist_ok=True)
    os.makedirs(gen_traj_dir, exist_ok=True)

    print('Writing reference PDB/XTC ...')
    _write_trajectory(ref_atom14, aatype, ref_traj_dir, args.protein)
    print('Writing generated PDB/XTC ...')
    _write_trajectory(gen_atom14, aatype, gen_traj_dir, args.protein)

    # --- Structural validity (STAR-MD paper metrics) ---
    print('Computing structural validity ...')
    validity = compute_validity(gen_atom14, aatype)
    print(f'  CA valid: {validity["ca_valid_pct"]:.1f}%  |  '
          f'AA valid: {validity["aa_valid_pct"]:.1f}%  |  '
          f'Combined: {validity["combined_valid_pct"]:.1f}%')

    # Save validity results
    import json as _json
    validity_path = os.path.join(args.out_dir, 'validity.json')
    with open(validity_path, 'w') as f:
        _json.dump(validity, f, indent=2)

    # --- Run MDGen analysis ---
    ref_parent = os.path.join(args.out_dir, 'ref')
    pkl_path = run_mdgen_analysis(
        ref_dir=ref_parent,
        gen_dir=gen_traj_dir,
        protein=args.protein,
        mode=args.mode,
        out_dir=args.out_dir,
        plot=not args.no_plot,
    )

    print_summary(pkl_path, args.mode, validity)


if __name__ == '__main__':
    main()
