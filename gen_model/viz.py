"""Trajectory visualization utilities.

Provides RMSD + 2D projection (matplotlib) and 3D CA spheres (py3Dmol)
for both .pt rigids trajectories and .pdb MDGen outputs.

Usage:
    from gen_model.viz import viz_trajectory_pt, viz_trajectory_pdb

    viz_trajectory_pt('outputs/conditional/4o66_C/traj.pt', title='4o66_C')
    viz_trajectory_pdb('outputs/mdgen/4o66_C.pdb', title='MDGen baseline')
"""

import os
import numpy as np


def viz_trajectory_pt(pt_path: str, title: str = ''):
    """Visualize a .pt trajectory ([T, N, 7] rigids): RMSD + 2D + py3Dmol."""
    import torch
    if not os.path.exists(pt_path):
        print(f'Not found: {pt_path}')
        return
    traj = torch.load(pt_path, map_location='cpu')
    ca = traj[:, :, 4:7].numpy()
    _viz_ca(ca, title)


def viz_trajectory_pdb(pdb_path: str, title: str = ''):
    """Visualize a .pdb trajectory (MDGen output): RMSD + 2D + py3Dmol."""
    if not os.path.exists(pdb_path):
        print(f'Not found: {pdb_path}')
        return
    import mdtraj
    t = mdtraj.load(pdb_path)
    ca = t.xyz * 10  # nm to Angstrom
    _viz_ca(ca, title)


def _viz_ca(ca: np.ndarray, title: str):
    """Shared CA visualization: RMSD plot + 2D projection + py3Dmol 3D.

    Args:
        ca: [T, N, 3] CA coordinates in Angstroms.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    T = len(ca)
    rmsd = np.sqrt(((ca - ca[0:1]) ** 2).sum(-1).mean(-1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(rmsd)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('RMSD')
    ax1.set_title('Drift from frame 0')

    ax2.plot(ca[0, :, 0], ca[0, :, 1], '-', lw=0.5, label='Frame 0')
    ax2.plot(ca[-1, :, 0], ca[-1, :, 1], '-', lw=0.5, label=f'Frame {T - 1}')
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.axis('off')
    ax2.set_title('First vs last (XY)')

    plt.suptitle(title or f'{T} frames')
    plt.tight_layout()
    plt.show()

    # 3D visualization
    try:
        import py3Dmol

        def _pdb(coords):
            return '\n'.join(
                f'ATOM  {j + 1:5d}  CA  ALA A{j + 1:4d}    '
                f'{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00'
                for j, x in enumerate(coords)
            )

        view = py3Dmol.view(width=600, height=400)
        view.addModel(_pdb(ca[0]), 'pdb')
        view.setStyle({'model': 0}, {'sphere': {'radius': 0.3, 'color': 'blue'}})
        view.addModel(_pdb(ca[-1]), 'pdb')
        view.setStyle({'model': 1}, {'sphere': {'radius': 0.3, 'color': 'orange'}})
        view.zoomTo()
        print(f'Blue=frame 0, Orange=frame {T - 1}')
        view.show()
    except ImportError:
        print('pip install py3Dmol for 3D visualization')
