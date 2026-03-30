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

# Patch mdgen's batched_gather for numpy ≥1.24 compatibility.
# Newer numpy requires tuple-indexing; list-indexing raises ValueError
# ("inhomogeneous shape") when the index list contains mixed types
# (torch tensors + slice objects).
import torch as _torch
import mdgen.tensor_utils as _mtu
import mdgen.geometry as _mg
def _batched_gather_fixed(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = _torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)
    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[tuple(ranges)]
_mtu.batched_gather = _batched_gather_fixed
_mg.batched_gather = _batched_gather_fixed

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
# Run MDGen analysis (pyemma) or fallback (deeptime + mdtraj)
# ---------------------------------------------------------------------------

def _has_pyemma() -> bool:
    try:
        import pyemma  # noqa: F401
        return True
    except ImportError:
        return False


def _compute_torsions(pdb_path: str, xtc_path: str, cossin: bool = False):
    """Extract backbone + sidechain torsion angles via mdtraj."""
    import mdtraj
    traj = mdtraj.load(xtc_path, top=pdb_path)
    _, phi = mdtraj.compute_phi(traj)
    _, psi = mdtraj.compute_psi(traj)
    _, omega = mdtraj.compute_omega(traj)
    _, chi1 = mdtraj.compute_chi1(traj)
    _, chi2 = mdtraj.compute_chi2(traj)

    labels = ([f'PHI_{i}' for i in range(phi.shape[1])]
              + [f'PSI_{i}' for i in range(psi.shape[1])]
              + [f'OMEGA_{i}' for i in range(omega.shape[1])]
              + [f'CHI1_{i}' for i in range(chi1.shape[1])]
              + [f'CHI2_{i}' for i in range(chi2.shape[1])])
    angles = np.concatenate([phi, psi, omega, chi1, chi2], axis=1)

    if cossin:
        cos_sin = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
        cs_labels = [f'COS({l})' for l in labels] + [f'SIN({l})' for l in labels]
        return cos_sin, cs_labels

    return angles, labels


def run_analysis_deeptime(ref_dir: str, gen_dir: str, protein: str,
                          mode: str = 'conditional',
                          compute_decorr: bool = True) -> dict:
    """Compute torsion JSD, TICA JSD, and decorrelation using deeptime + mdtraj.

    Fallback for when pyemma is not installed. Produces the same result
    structure as MDGen's analyze_peptide_sim.py.
    """
    from scipy.spatial.distance import jensenshannon
    from deeptime.decomposition import TICA

    ref_pdb = os.path.join(ref_dir, protein, f'{protein}.pdb')
    ref_xtc = os.path.join(ref_dir, protein, f'{protein}.xtc')
    gen_pdb = os.path.join(gen_dir, f'{protein}.pdb')
    gen_xtc = os.path.join(gen_dir, f'{protein}.xtc')

    results = {'JSD': {}}

    # --- Torsion JSD (raw angles) ---
    ref_ang, labels = _compute_torsions(ref_pdb, ref_xtc, cossin=False)
    gen_ang, _      = _compute_torsions(gen_pdb, gen_xtc, cossin=False)

    # The generated trajectory has approximate backbone only (no sidechain beyond
    # CB), so mdtraj may compute fewer chi1/chi2 features. Clamp to the minimum.
    n_feats = min(ref_ang.shape[1], gen_ang.shape[1])
    labels  = labels[:n_feats]

    for i, feat in enumerate(labels):
        ref_p = np.histogram(ref_ang[:, i], range=(-np.pi, np.pi), bins=100)[0]
        gen_p = np.histogram(gen_ang[:, i], range=(-np.pi, np.pi), bins=100)[0]
        results['JSD'][feat] = float(jensenshannon(ref_p, gen_p))

    # 2D phi-psi JSD
    phi_count = sum(1 for l in labels if l.startswith('PHI_'))
    psi_count = sum(1 for l in labels if l.startswith('PSI_'))
    n_pairs = min(phi_count, psi_count)
    for i in range(n_pairs):
        phi_idx = i               # PHI_i
        psi_idx = phi_count + i   # PSI_i
        r = ((-np.pi, np.pi), (-np.pi, np.pi))
        ref_p = np.histogram2d(ref_ang[:, phi_idx], ref_ang[:, psi_idx],
                               range=r, bins=50)[0]
        gen_p = np.histogram2d(gen_ang[:, phi_idx], gen_ang[:, psi_idx],
                               range=r, bins=50)[0]
        pair_label = f'{labels[phi_idx]}|{labels[psi_idx]}'
        results['JSD'][pair_label] = float(jensenshannon(ref_p.flatten(),
                                                          gen_p.flatten()))

    # --- TICA JSD (cos/sin features, deeptime TICA) ---
    ref_cs, _ = _compute_torsions(ref_pdb, ref_xtc, cossin=True)
    gen_cs, _ = _compute_torsions(gen_pdb, gen_xtc, cossin=True)
    # Align feature count (same root cause as raw-angle clamp above)
    n_cs = min(ref_cs.shape[1], gen_cs.shape[1])
    ref_cs, gen_cs = ref_cs[:, :n_cs], gen_cs[:, :n_cs]

    lag = min(1000, ref_cs.shape[0] // 3)
    tica = TICA(lagtime=lag, dim=2).fit_fetch(ref_cs)
    ref_tica = tica.transform(ref_cs)
    gen_tica = tica.transform(gen_cs)

    t0_min = min(ref_tica[:, 0].min(), gen_tica[:, 0].min())
    t0_max = max(ref_tica[:, 0].max(), gen_tica[:, 0].max())
    t1_min = min(ref_tica[:, 1].min(), gen_tica[:, 1].min())
    t1_max = max(ref_tica[:, 1].max(), gen_tica[:, 1].max())

    ref_p = np.histogram(ref_tica[:, 0], range=(t0_min, t0_max), bins=100)[0]
    gen_p = np.histogram(gen_tica[:, 0], range=(t0_min, t0_max), bins=100)[0]
    results['JSD']['TICA-0'] = float(jensenshannon(ref_p, gen_p))

    ref_p = np.histogram2d(*ref_tica[:, :2].T,
                           range=((t0_min, t0_max), (t1_min, t1_max)),
                           bins=50)[0]
    gen_p = np.histogram2d(*gen_tica[:, :2].T,
                           range=((t0_min, t0_max), (t1_min, t1_max)),
                           bins=50)[0]
    results['JSD']['TICA-0,1'] = float(jensenshannon(ref_p.flatten(),
                                                       gen_p.flatten()))

    # Store TICA projections for plotting
    results['ref_tica'] = ref_tica[:, :2].astype(np.float32)
    results['gen_tica'] = gen_tica[:, :2].astype(np.float32)

    # --- Decorrelation (torsion + TICA autocorrelation) ---
    if compute_decorr and mode == 'conditional':
        from statsmodels.tsa.stattools import acovf

        nlag_ref = min(100_000, ref_ang.shape[0] - 1)
        nlag_gen = min(1000, gen_ang.shape[0] - 1)

        results['md_decorrelation'] = {}
        results['our_decorrelation'] = {}
        for i, feat in enumerate(labels):
            for src, data, nl, key in [
                ('md', ref_ang, nlag_ref, 'md_decorrelation'),
                ('our', gen_ang, nlag_gen, 'our_decorrelation'),
            ]:
                ac = (acovf(np.sin(data[:, i]), demean=False, adjusted=True, nlag=nl)
                      + acovf(np.cos(data[:, i]), demean=False, adjusted=True, nlag=nl))
                baseline = np.sin(data[:, i]).mean()**2 + np.cos(data[:, i]).mean()**2
                results[key][feat] = ((ac - baseline) / (1 - baseline + 1e-8)).astype(np.float16)

        # TICA decorrelation
        ac_ref = acovf(ref_tica[:, 0], nlag=nlag_ref, adjusted=True, demean=False)
        results['md_decorrelation']['tica'] = ac_ref.astype(np.float16)
        ac_gen = acovf(gen_tica[:, 0], nlag=nlag_gen, adjusted=True, demean=False)
        results['our_decorrelation']['tica'] = ac_gen.astype(np.float16)

    return results


def run_mdgen_analysis(ref_dir: str, gen_dir: str, protein: str,
                       mode: str = 'conditional',
                       out_dir: str = 'outputs/eval',
                       plot: bool = True) -> str:
    """Run evaluation metrics: tries pyemma (MDGen script), falls back to deeptime.

    Returns:
        Path to the saved .pkl results file.
    """
    import pickle

    save_name = 'eval_results.pkl'
    pkl_path = os.path.join(gen_dir, save_name)

    if _has_pyemma():
        script = os.path.join(
            os.path.dirname(__file__), '..', 'extern', 'mdgen', 'scripts',
            'analyze_peptide_sim.py')
        script = os.path.abspath(script)

        cmd = [
            sys.executable, script,
            '--mddir', ref_dir,
            '--pdbdir', gen_dir,
            '--pdb_id', protein,
            '--save',
            '--save_name', save_name,
            '--no_msm',
        ]
        if mode == 'unconditional':
            cmd.append('--no_decorr')
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
    else:
        print('pyemma not available — using deeptime + mdtraj fallback')
        results = run_analysis_deeptime(
            ref_dir, gen_dir, protein, mode=mode,
            compute_decorr=(mode == 'conditional'))
        all_results = {protein: results}
        with open(pkl_path, 'wb') as f:
            pickle.dump(all_results, f)

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


def plot_summary(pkl_path: str, out_dir: str, protein: str, mode: str):
    """Generate and save evaluation plots from the results pickle.

    Produces two figures saved as PNGs in out_dir:
      eval_tica.png   — TICA 1D histogram + 2D landscape (ref vs gen) + JSD bar chart
      eval_autocorr.png — mean backbone torsion autocorrelation + TICA-0 autocorrelation
    """
    import pickle
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    with open(pkl_path, 'rb') as f:
        all_results = pickle.load(f)

    if protein not in all_results:
        print(f'plot_summary: protein {protein} not in results')
        return

    results = all_results[protein]
    jsd = results.get('JSD', {})

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Figure 1: TICA landscapes + JSD bar chart                          #
    # ------------------------------------------------------------------ #
    has_tica = 'ref_tica' in results and 'gen_tica' in results
    fig1, axes = plt.subplots(1, 4, figsize=(18, 4))

    if has_tica:
        ref_t = results['ref_tica']   # [T_ref, 2]
        gen_t = results['gen_tica']   # [T_gen, 2]

        t0_min = min(ref_t[:, 0].min(), gen_t[:, 0].min())
        t0_max = max(ref_t[:, 0].max(), gen_t[:, 0].max())
        t1_min = min(ref_t[:, 1].min(), gen_t[:, 1].min())
        t1_max = max(ref_t[:, 1].max(), gen_t[:, 1].max())
        bins1d  = np.linspace(t0_min, t0_max, 80)

        # 1D TICA-0 distribution
        ax = axes[0]
        ax.hist(ref_t[:, 0], bins=bins1d, density=True,
                alpha=0.6, color='steelblue', label='Reference')
        ax.hist(gen_t[:, 0], bins=bins1d, density=True,
                alpha=0.6, color='tomato',   label='Generated')
        ax.set_xlabel('TICA-0')
        ax.set_ylabel('Density')
        ax.set_title(f'TICA-0  (JSD={jsd.get("TICA-0", float("nan")):.3f})')
        ax.legend(fontsize=8)

        # 2D free-energy landscape helper
        def _fes(coords, t0_r, t1_r, bins=50):
            H, xe, ye = np.histogram2d(coords[:, 0], coords[:, 1],
                                       range=[[t0_r[0], t0_r[1]],
                                              [t1_r[0], t1_r[1]]],
                                       bins=bins, density=True)
            H = np.maximum(H, 1e-10)
            fes = -np.log(H)
            fes -= fes.min()
            cx = 0.5 * (xe[:-1] + xe[1:])
            cy = 0.5 * (ye[:-1] + ye[1:])
            return fes.T, cx, cy

        t0_r = (t0_min, t0_max)
        t1_r = (t1_min, t1_max)

        fes_ref, cx, cy = _fes(ref_t, t0_r, t1_r)
        fes_gen, _,  _  = _fes(gen_t, t0_r, t1_r)
        vmax = max(fes_ref.max(), fes_gen.max())

        for ax, fes, label in [(axes[1], fes_ref, 'Reference'),
                               (axes[2], fes_gen, 'Generated')]:
            im = ax.contourf(cx, cy, fes, levels=20,
                             cmap='RdYlBu_r', vmin=0, vmax=vmax)
            ax.set_xlabel('TICA-0')
            ax.set_ylabel('TICA-1')
            ax.set_title(f'FES — {label}')
            plt.colorbar(im, ax=ax, label='–log p (kT)')
    else:
        for ax in axes[:3]:
            ax.text(0.5, 0.5, 'TICA not available', ha='center', va='center',
                    transform=ax.transAxes)

    # JSD bar chart by torsion group
    groups = {
        'PHI': [v for k, v in jsd.items() if k.startswith('PHI_')],
        'PSI': [v for k, v in jsd.items() if k.startswith('PSI_')],
        'OMEGA': [v for k, v in jsd.items() if k.startswith('OMEGA_')],
        'CHI1': [v for k, v in jsd.items() if k.startswith('CHI1_')],
        'CHI2': [v for k, v in jsd.items() if k.startswith('CHI2_')],
        'TICA': [v for k, v in jsd.items() if k.startswith('TICA')],
    }
    group_means = {k: float(np.mean(v)) for k, v in groups.items() if v}
    ax = axes[3]
    bars = ax.bar(group_means.keys(), group_means.values(),
                  color=['#4878cf', '#6acc65', '#d65f5f',
                         '#b47cc7', '#c4ad66', '#77bedb'][:len(group_means)])
    ax.set_ylabel('Mean JSD')
    ax.set_title('Torsion JSD by group')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, group_means.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    fig1.suptitle(f'{protein} — TICA & JSD  ({mode})', fontsize=12)
    fig1.tight_layout()
    tica_path = os.path.join(out_dir, 'eval_tica.png')
    fig1.savefig(tica_path, dpi=120, bbox_inches='tight')
    plt.close(fig1)
    print(f'Saved TICA plot → {tica_path}')

    # ------------------------------------------------------------------ #
    # Figure 2: Autocorrelation                                          #
    # ------------------------------------------------------------------ #
    if mode != 'conditional':
        return tica_path

    md_dec  = results.get('md_decorrelation',  {})
    our_dec = results.get('our_decorrelation', {})
    if not md_dec or not our_dec:
        return tica_path

    fig2, (ax_bb, ax_tica) = plt.subplots(1, 2, figsize=(12, 4))

    # Mean backbone torsion autocorrelation
    bb_keys = [k for k in md_dec if k != 'tica' and
               any(k.startswith(p) for p in ('PHI_', 'PSI_', 'OMEGA_'))]

    if bb_keys:
        ref_bb = np.array([md_dec[k].astype(np.float32) for k in bb_keys])
        gen_bb = np.array([our_dec[k].astype(np.float32)
                           for k in bb_keys if k in our_dec])

        ref_mean = ref_bb.mean(axis=0)
        gen_mean = gen_bb.mean(axis=0) if gen_bb.size else None

        # Subsample lags for readability
        max_show = 500
        ref_lags = np.arange(len(ref_mean))
        stride_r = max(1, len(ref_lags) // max_show)
        ax_bb.plot(ref_lags[::stride_r], ref_mean[::stride_r],
                   color='steelblue', lw=1.2, label='Reference MD')
        if gen_mean is not None:
            gen_lags = np.arange(len(gen_mean))
            stride_g = max(1, len(gen_lags) // max_show)
            ax_bb.plot(gen_lags[::stride_g], gen_mean[::stride_g],
                       color='tomato', lw=1.2, label='Generated')
        ax_bb.axhline(0, color='k', lw=0.5, ls='--')
        ax_bb.set_xlabel('Lag (frames)')
        ax_bb.set_ylabel('Autocorrelation')
        ax_bb.set_title('Mean backbone torsion autocorrelation')
        ax_bb.legend()
        ax_bb.grid(True, alpha=0.3)

    # TICA-0 autocorrelation
    if 'tica' in md_dec:
        ref_ac = md_dec['tica'].astype(np.float32)
        gen_ac = our_dec.get('tica', np.array([])).astype(np.float32)

        ref_lags = np.arange(len(ref_ac))
        stride_r = max(1, len(ref_lags) // max_show)
        ax_tica.plot(ref_lags[::stride_r], ref_ac[::stride_r],
                     color='steelblue', lw=1.2, label='Reference MD')
        if gen_ac.size:
            gen_lags = np.arange(len(gen_ac))
            stride_g = max(1, len(gen_lags) // max_show)
            ax_tica.plot(gen_lags[::stride_g], gen_ac[::stride_g],
                         color='tomato', lw=1.2, label='Generated')
        ax_tica.axhline(0, color='k', lw=0.5, ls='--')
        ax_tica.set_xlabel('Lag (frames)')
        ax_tica.set_ylabel('Autocorrelation')
        ax_tica.set_title('TICA-0 autocorrelation')
        ax_tica.legend()
        ax_tica.grid(True, alpha=0.3)

    fig2.suptitle(f'{protein} — Autocorrelation  ({mode})', fontsize=12)
    fig2.tight_layout()
    autocorr_path = os.path.join(out_dir, 'eval_autocorr.png')
    fig2.savefig(autocorr_path, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved autocorrelation plot → {autocorr_path}')

    return tica_path


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
    parser.add_argument('--max_ref_frames', type=int, default=2000,
                        help='Max reference frames to write to PDB/XTC '
                             '(evenly subsampled). 0 = no limit.')
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

    # Subsample reference to avoid writing 30k-frame PDB files
    T_ref = ref_atom14.shape[0]
    if args.max_ref_frames > 0 and T_ref > args.max_ref_frames:
        idx = np.round(np.linspace(0, T_ref - 1, args.max_ref_frames)).astype(int)
        ref_atom14 = ref_atom14[idx]
        print(f'  Subsampled reference: {T_ref} → {ref_atom14.shape[0]} frames')

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

    if not args.no_plot:
        plot_summary(pkl_path, gen_traj_dir, args.protein, args.mode)


if __name__ == '__main__':
    main()
