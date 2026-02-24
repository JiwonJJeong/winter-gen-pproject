"""Utilities for converting SE(3) Rigid objects to atom-level protein structures.

All three rigids_to_* functions accept either a single Rigid [N] or a list of
Rigids [N] (as returned by the inference and video_extrapolation pipelines) and
return numpy arrays in Angstroms.

Two helpers are provided for visualisation:

    save_ca_trajectory_pdb(ca_traj, out_path, seqres)
        Writes a multi-model PDB file from a [T, N, 3] CA array.
        Opens directly in PyMOL, VMD, or UCSF ChimeraX and animates as a
        trajectory.

    as_mdtraj_trajectory(ca_traj, seqres) -> mdtraj.Trajectory
        Wraps the CA array in an MDTraj Trajectory object (coordinates are
        converted from Angstroms to nm).  From there you can:
            traj.save_pdb('out.pdb')
            traj.save_dcd('out.dcd')
            import nglview; nglview.show_mdtraj(traj)   # Jupyter inline

Example usage:
    from gen_model.utils.structure_utils import (
        rigids_to_ca_trajectory,
        rigids_to_atom14_trajectory,
        rigids_to_atom37_trajectory,
        save_ca_trajectory_pdb,
        as_mdtraj_trajectory,
    )

    # From video_extrapolation.build_trajectory:
    ca   = rigids_to_ca_trajectory(frames, coord_scale)              # [T, N, 3]
    bb   = rigids_to_atom37_trajectory(frames, aatype, coord_scale)  # [T, N, 37, 3]

    # Visualise the CA trajectory:
    save_ca_trajectory_pdb(ca, 'traj.pdb', seqres)          # open in PyMOL/VMD
    traj = as_mdtraj_trajectory(ca, seqres)
    traj.save_dcd('traj.dcd')                                # VMD / MDAnalysis
    import nglview; nglview.show_mdtraj(traj)               # Jupyter inline
"""
import numpy as np
import torch

from gen_model.utils.rigid_utils import Rigid
from gen_model.data.geometry import frames_torsions_to_atom14, frames_torsions_to_atom37
from gen_model.data.residue_constants import restype_1to3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_list(rigids):
    """Accept a single Rigid or list/tuple of Rigids; always return a list."""
    if isinstance(rigids, (list, tuple)):
        return list(rigids)
    return [rigids]


def _unscale_rigid(rigid: Rigid, coord_scale: float) -> Rigid:
    """Return a new Rigid with translations divided by coord_scale (→ Angstroms)."""
    return Rigid(
        rots=rigid.get_rots(),
        trans=rigid.get_trans().detach().cpu().float() / coord_scale,
    )


def _make_torsions_list(torsions, T: int, N: int):
    """Normalise the torsions argument into a list of T tensors [N, 7, 2]."""
    if torsions is None:
        return [torch.zeros(N, 7, 2)] * T
    if isinstance(torsions, (list, tuple)):
        return [
            t if isinstance(t, torch.Tensor) else torch.from_numpy(np.asarray(t, dtype=np.float32))
            for t in torsions
        ]
    # Single array/tensor — broadcast to every frame
    t = torsions if isinstance(torsions, torch.Tensor) else torch.from_numpy(
        np.asarray(torsions, dtype=np.float32)
    )
    return [t] * T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rigids_to_ca_trajectory(
    rigids,
    coord_scale: float,
) -> np.ndarray:
    """Extract Cα positions from one or more Rigid objects.

    Args:
        rigids:      Single Rigid [N] or list of Rigid [N].
        coord_scale: 1/std scaling applied during training.
                     Translations are divided by this to recover Angstroms.

    Returns:
        [T, N, 3] float32 numpy array in Angstroms, centred at origin.
        T=1 when a single Rigid is passed.
    """
    frames = _ensure_list(rigids)
    ca = np.stack([
        r.get_trans().detach().cpu().float().numpy() / coord_scale
        for r in frames
    ], axis=0)                                   # [T, N, 3]
    return ca


def rigids_to_atom14_trajectory(
    rigids,
    aatype,
    coord_scale: float,
    torsions=None,
) -> np.ndarray:
    """Convert Rigid objects to atom14 heavy-atom positions.

    Args:
        rigids:      Single Rigid [N] or list of Rigid [N].
        aatype:      [N] residue type indices (int tensor or numpy array).
        coord_scale: Scale factor to convert translations to Angstroms.
        torsions:    [N, 7, 2] sin/cos torsion angles, a list of per-frame
                     [N, 7, 2] arrays, or None.  When None, zeros are used —
                     this reconstructs backbone geometry correctly; side chains
                     will be in an extended (unphysical) conformation.

    Returns:
        [T, N, 14, 3] float32 numpy array in Angstroms.
    """
    frames = _ensure_list(rigids)
    T = len(frames)

    if not isinstance(aatype, torch.Tensor):
        aatype = torch.tensor(np.asarray(aatype), dtype=torch.long)

    tors_list = _make_torsions_list(torsions, T, aatype.shape[0])

    out = []
    for rigid, tors in zip(frames, tors_list):
        atom14 = frames_torsions_to_atom14(
            _unscale_rigid(rigid, coord_scale), tors, aatype
        )
        out.append(atom14.detach().cpu().float().numpy())

    return np.stack(out, axis=0)                 # [T, N, 14, 3]


def rigids_to_atom37_trajectory(
    rigids,
    aatype,
    coord_scale: float,
    torsions=None,
) -> np.ndarray:
    """Convert Rigid objects to atom37 heavy-atom positions.

    Args:
        rigids:      Single Rigid [N] or list of Rigid [N].
        aatype:      [N] residue type indices (int tensor or numpy array).
        coord_scale: Scale factor to convert translations to Angstroms.
        torsions:    [N, 7, 2] sin/cos torsion angles, a list of per-frame
                     [N, 7, 2] arrays, or None.  When None, zeros are used —
                     backbone geometry is correct; side chains will be in an
                     extended (unphysical) conformation.

    Returns:
        [T, N, 37, 3] float32 numpy array in Angstroms.
    """
    frames = _ensure_list(rigids)
    T = len(frames)

    if not isinstance(aatype, torch.Tensor):
        aatype = torch.tensor(np.asarray(aatype), dtype=torch.long)

    tors_list = _make_torsions_list(torsions, T, aatype.shape[0])

    out = []
    for rigid, tors in zip(frames, tors_list):
        atom37 = frames_torsions_to_atom37(
            _unscale_rigid(rigid, coord_scale), tors, aatype
        )
        out.append(atom37.detach().cpu().float().numpy())

    return np.stack(out, axis=0)                 # [T, N, 37, 3]


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def save_ca_trajectory_pdb(
    ca_traj: np.ndarray,
    out_path: str,
    seqres: str,
) -> None:
    """Write a CA-only trajectory as a multi-model PDB file.

    Args:
        ca_traj:  [T, N, 3] float32 array in Angstroms.
        out_path: Destination .pdb file path.
        seqres:   Single-letter amino-acid sequence of length N.

    The file opens directly in PyMOL, VMD, or UCSF ChimeraX and plays back as
    an animated trajectory.  No extra Python dependencies are required.
    """
    T, N, _ = ca_traj.shape
    res3_list = [restype_1to3.get(aa, 'GLY') for aa in seqres]

    with open(out_path, 'w') as fh:
        for t in range(T):
            fh.write(f'MODEL     {t + 1:4d}\n')
            for i in range(N):
                x, y, z = ca_traj[t, i]
                fh.write(
                    f'ATOM  {i + 1:5d}  CA  {res3_list[i]:3s} A{i + 1:4d}    '
                    f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n'
                )
            fh.write('ENDMDL\n')
        fh.write('END\n')


def as_mdtraj_trajectory(
    ca_traj: np.ndarray,
    seqres: str,
):
    """Wrap a [T, N, 3] CA array in an mdtraj.Trajectory.

    Coordinates are converted from Angstroms to nanometres (MDTraj convention).

    Args:
        ca_traj: [T, N, 3] float32 array in Angstroms.
        seqres:  Single-letter amino-acid sequence of length N.

    Returns:
        mdtraj.Trajectory with a CA-only topology.

    Example:
        traj = as_mdtraj_trajectory(ca, seqres)
        traj.save_pdb('out.pdb')
        traj.save_dcd('out.dcd')                    # compact binary
        import nglview; nglview.show_mdtraj(traj)   # Jupyter inline 3-D viewer
    """
    import mdtraj as md

    topology = md.Topology()
    chain = topology.add_chain()
    for aa in seqres:
        residue = topology.add_residue(restype_1to3.get(aa, 'GLY'), chain)
        topology.add_atom('CA', md.element.carbon, residue)

    xyz_nm = ca_traj.astype(np.float32) / 10.0   # Å → nm (MDTraj convention)
    return md.Trajectory(xyz_nm, topology)
