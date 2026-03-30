"""Trajectory visualization utilities.

Provides RMSD + 2D trajectory overlay (matplotlib) and an animated
3D backbone-stick viewer (py3Dmol) for both .pt rigids trajectories
and .pdb MDGen outputs.

Usage:
    from gen_model.viz import viz_trajectory_pt, viz_trajectory_pdb

    viz_trajectory_pt('outputs/conditional/4o66_C/traj.pt', title='4o66_C')
    viz_trajectory_pdb('outputs/mdgen/4o66_C.pdb', title='MDGen baseline')
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def viz_trajectory_pt(pt_path: str, title: str = ''):
    """Visualize a .pt trajectory ([T, N, 7] rigids): RMSD + 2D + animated 3D."""
    import torch
    if not os.path.exists(pt_path):
        print(f'Not found: {pt_path}')
        return
    traj = torch.load(pt_path, map_location='cpu')
    ca = traj[:, :, 4:7].numpy()   # translations as CA proxy, [T, N, 3]
    _viz_ca(ca, title)


def viz_trajectory_pdb(pdb_path: str, title: str = ''):
    """Visualize a multi-model .pdb trajectory: RMSD + 2D + animated 3D."""
    if not os.path.exists(pdb_path):
        print(f'Not found: {pdb_path}')
        return
    import mdtraj
    t = mdtraj.load(pdb_path)
    ca_idx = t.topology.select('name CA')
    ca = t.xyz[:, ca_idx, :] * 10  # nm → Å, [T, N, 3]
    _viz_ca(ca, title)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_multimodel_pdb(ca_frames: list) -> str:
    """Build a multi-model PDB string with CA atoms and backbone CONECT records.

    Each entry in ca_frames is [N, 3].  CONECT records connect consecutive
    CA atoms so py3Dmol can render them as a chain (stick/line style).
    """
    lines = []
    N = ca_frames[0].shape[0]
    for mi, ca in enumerate(ca_frames):
        lines.append(f'MODEL     {mi + 1:4d}')
        for j, (x, y, z) in enumerate(ca):
            lines.append(
                f'ATOM  {j + 1:5d}  CA  ALA A{j + 1:4d}    '
                f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  '
            )
        # CONECT records: CA_j — CA_{j+1} for chain connectivity
        for j in range(1, N):
            lines.append(f'CONECT{j:5d}{j + 1:5d}')
        lines.append('ENDMDL')
    return '\n'.join(lines)


def _viz_ca(ca: np.ndarray, title: str):
    """Shared CA visualization: RMSD plot + 2D trajectory overlay + animated 3D.

    Args:
        ca: [T, N, 3] CA coordinates in Angstroms.
        title: Plot title.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    T, N, _ = ca.shape

    # --- Centre each frame (remove rigid translation drift for display) ------
    ca_centred = ca - ca.mean(axis=1, keepdims=True)   # [T, N, 3]

    # RMSD from frame 0 (after centering)
    rmsd = np.sqrt(((ca_centred - ca_centred[0:1]) ** 2).sum(-1).mean(-1))

    # --- Plot ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Left: RMSD over time
    ax1.plot(rmsd, lw=1.0)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('RMSD (Å)')
    ax1.set_title('RMSD from frame 0')
    ax1.grid(True, alpha=0.3)

    # Right: 2D trajectory overlay (XY, colour = time: blue → red)
    # Show ~30 evenly-spaced frames to keep the plot readable
    n_show = min(T, 30)
    idx = np.round(np.linspace(0, T - 1, n_show)).astype(int)
    cmap = cm.coolwarm
    for k, t_idx in enumerate(idx):
        color = cmap(k / max(n_show - 1, 1))
        alpha = 0.4 + 0.6 * k / max(n_show - 1, 1)
        ax2.plot(ca_centred[t_idx, :, 0], ca_centred[t_idx, :, 1],
                 '-o', lw=0.8, ms=1.5, color=color, alpha=alpha)

    # Colour-bar legend
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=T - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax2, label='Frame index', fraction=0.046, pad=0.04)
    ax2.set_aspect('equal')
    ax2.set_title('CA chain — trajectory (blue→red)')
    ax2.axis('off')

    plt.suptitle(title or f'{T} frames, {N} residues')
    plt.tight_layout()
    plt.show()

    # --- Animated 3D via py3Dmol --------------------------------------------
    try:
        import py3Dmol

        # Subsample: target ~60 frames for smooth animation without lag
        n_anim = min(T, 60)
        anim_idx = np.round(np.linspace(0, T - 1, n_anim)).astype(int)
        anim_frames = [ca_centred[i] for i in anim_idx]

        pdb_str = _build_multimodel_pdb(anim_frames)

        view = py3Dmol.view(width=720, height=480)
        view.addModelsAsFrames(pdb_str, 'pdb')
        # Backbone sticks + CA spheres — gives a chain appearance without
        # needing N/C atoms (those require full-atom reconstruction)
        view.setStyle({}, {
            'stick': {'radius': 0.12, 'color': 'spectrum'},
            'sphere': {'radius': 0.25, 'color': 'spectrum'},
        })
        view.setBackgroundColor('0xf0f0f0')
        view.zoomTo()
        # Loop animation at ~80 ms/frame
        view.animate({'loop': 'forward', 'interval': 80, 'reps': 0})
        print(f'Animated {n_anim} frames ({T} total) | '
              f'colour gradient = N-terminus (blue) → C-terminus (red)')
        view.show()

    except ImportError:
        print('pip install py3Dmol for 3D visualization')
