"""Utilities for converting SE(3) Rigid objects to atom-level protein structures.

All three rigids_to_* functions accept either a single Rigid [N] or a list of
Rigids [N] (as returned by the inference and video_extrapolation pipelines) and
return numpy arrays in Angstroms.

Visualisation helpers
---------------------

    aatype_to_seqres(aatype) -> str
        Converts an integer aatype array (as stored in dataset batches) back to
        a single-letter sequence string.  Use this to avoid re-reading the CSV
        when seqres is needed for downstream visualisation.

    save_ca_trajectory_pdb(ca_traj, out_path, seqres)
        Writes a multi-model PDB file from a [T, N, 3] CA array.
        Opens directly in PyMOL, VMD, or UCSF ChimeraX and animates as a
        trajectory.  seqres may be a str or an aatype int array/tensor.

    as_mdtraj_trajectory(ca_traj, seqres) -> mdtraj.Trajectory
        Wraps the CA array in an MDTraj Trajectory object (coordinates are
        converted from Angstroms to nm).  seqres may be str or aatype.

    show_py3dmol_trajectory(ca_traj, seqres, *, color, ...) -> py3Dmol.view
        Renders a [T, N, 3] CA array as an inline animated py3Dmol widget.
        Returns the view so the caller can call .show() or chain further styling.

    show_py3dmol_overlay(structures, seqres, *, colors, ...) -> py3Dmol.view
        Overlays multiple single-frame CA structures in one py3Dmol view.
        Useful for side-by-side comparison of real vs generated conformations.

Example usage
-------------
    from gen_model.utils.structure_utils import (
        aatype_to_seqres,
        rigids_to_ca_trajectory,
        show_py3dmol_trajectory,
        show_py3dmol_overlay,
        save_ca_trajectory_pdb,
    )

    # aatype comes from the dataset batch — no CSV lookup needed:
    seqres = aatype_to_seqres(batch['aatype'])

    ca  = rigids_to_ca_trajectory(frames, coord_scale)   # [T, N, 3]

    # Animate the full trajectory:
    view = show_py3dmol_trajectory(ca, seqres, color='blue')
    view.show()

    # Compare a reference frame (blue) to a generated frame (orange):
    view = show_py3dmol_overlay([ref_ca[0], gen_ca[0]], seqres)
    view.show()

    # Save to PDB (seqres or aatype both accepted):
    save_ca_trajectory_pdb(ca, 'traj.pdb', seqres)
"""
import io
import numpy as np
import torch

from gen_model.utils.rigid_utils import Rigid
from gen_model.data.geometry import frames_torsions_to_atom14, frames_torsions_to_atom37
from gen_model.data.residue_constants import restype_1to3, restypes_with_x


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


def _coerce_seqres(seqres_or_aatype) -> str:
    """Accept str or int array/tensor; always return a seqres string."""
    if isinstance(seqres_or_aatype, str):
        return seqres_or_aatype
    return aatype_to_seqres(seqres_or_aatype)


def _build_pdb_string(ca_traj: np.ndarray, seqres: str) -> str:
    """Build a multi-model PDB string in memory (no file I/O).

    Args:
        ca_traj: [T, N, 3] float32 array in Angstroms.
        seqres:  Single-letter sequence of length N (already coerced to str).
    """
    T, N, _ = ca_traj.shape
    res3_list = [restype_1to3.get(aa, 'GLY') for aa in seqres]
    buf = io.StringIO()
    # Build CONECT block once (same serial numbers for every frame).
    # Must be inside each MODEL block so addModelsAsFrames picks them up.
    conect_lines = ''.join(
        f'CONECT{i + 1:5d}{i + 2:5d}\n' for i in range(N - 1)
    )
    for t in range(T):
        buf.write(f'MODEL     {t + 1:4d}\n')
        for i in range(N):
            x, y, z = ca_traj[t, i]
            buf.write(
                f'ATOM  {i + 1:5d}  CA  {res3_list[i]:3s} A{i + 1:4d}    '
                f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n'
            )
        buf.write(conect_lines)
        buf.write('ENDMDL\n')
    buf.write('END\n')
    return buf.getvalue()


# Default sphere highlight colours (the "pearls" on the Cα trace)
_SPHERE_DEFAULTS = {
    'blue':   'lightblue',
    'orange': '#FFB347',
    'red':    '#FF9999',
    'green':  '#90EE90',
    'purple': '#DDA0DD',
    'cyan':   '#AFEEEE',
}
_DEFAULT_COLOR_CYCLE = ['blue', 'orange', 'green', 'red', 'purple', 'cyan']


# ---------------------------------------------------------------------------
# Public API — sequence recovery
# ---------------------------------------------------------------------------

def aatype_to_seqres(aatype) -> str:
    """Convert an integer aatype array to a single-letter sequence string.

    Avoids re-reading the atlas CSV when seqres is needed downstream.
    Unknown indices (≥ 20) are mapped to 'X'.

    Args:
        aatype: [N] int tensor or numpy array of residue-type indices
                (as stored in dataset batch['aatype']).

    Returns:
        Single-letter amino-acid sequence string of length N.

    Example:
        # In an inference notebook — aatype was already used for the model:
        seqres = aatype_to_seqres(batch['aatype'])
        view = show_py3dmol_trajectory(ca, seqres, color='blue')
        view.show()
    """
    if isinstance(aatype, torch.Tensor):
        indices = aatype.detach().cpu().tolist()
    else:
        indices = list(np.asarray(aatype).ravel())
    return ''.join(restypes_with_x[min(int(i), 20)] for i in indices)


# ---------------------------------------------------------------------------
# Public API — Rigid → coordinate arrays
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
    seqres,
) -> None:
    """Write a CA-only trajectory as a multi-model PDB file.

    Args:
        ca_traj:  [T, N, 3] float32 array in Angstroms.
        out_path: Destination .pdb file path.
        seqres:   Single-letter amino-acid sequence (str) or integer aatype
                  array/tensor of length N.

    The file opens directly in PyMOL, VMD, or UCSF ChimeraX and plays back as
    an animated trajectory.  No extra Python dependencies are required.
    """
    seqres = _coerce_seqres(seqres)
    with open(out_path, 'w') as fh:
        fh.write(_build_pdb_string(ca_traj, seqres))


def as_mdtraj_trajectory(
    ca_traj: np.ndarray,
    seqres,
):
    """Wrap a [T, N, 3] CA array in an mdtraj.Trajectory.

    Coordinates are converted from Angstroms to nanometres (MDTraj convention).

    Args:
        ca_traj: [T, N, 3] float32 array in Angstroms.
        seqres:  Single-letter amino-acid sequence (str) or integer aatype
                 array/tensor of length N.

    Returns:
        mdtraj.Trajectory with a CA-only topology.

    Example:
        traj = as_mdtraj_trajectory(ca, batch['aatype'])
        traj.save_pdb('out.pdb')
        traj.save_dcd('out.dcd')    # compact binary
    """
    import mdtraj as md

    seqres = _coerce_seqres(seqres)
    topology = md.Topology()
    chain = topology.add_chain()
    for aa in seqres:
        residue = topology.add_residue(restype_1to3.get(aa, 'GLY'), chain)
        topology.add_atom('CA', md.element.carbon, residue)

    xyz_nm = ca_traj.astype(np.float32) / 10.0   # Å → nm (MDTraj convention)
    return md.Trajectory(xyz_nm, topology)


def show_py3dmol_trajectory(
    ca_traj: np.ndarray,
    seqres,
    *,
    color: str = 'blue',
    sphere_color: str = None,
    radius: float = 0.3,
    sphere_radius: float = 0.3,
    animate: bool = True,
    interval: int = 150,
    width: int = 800,
    height: int = 400,
):
    """Render a CA trajectory as an inline animated py3Dmol view.

    The trajectory is shown as a Cα trace ("string") with small spheres at
    each Cα position ("pearls").  Call .show() on the returned view to display
    it inline in a Jupyter / Colab cell.

    Args:
        ca_traj:      [T, N, 3] or [N, 3] float32 array in Angstroms.
        seqres:       Sequence string (str) or integer aatype array/tensor [N].
        color:        Trace and sphere colour (any CSS colour string).
        sphere_color: Override the sphere colour.  Defaults to a lighter
                      tint of `color` (e.g. 'lightblue' for 'blue').
        radius:       Trace tube radius in Ångströms.
        sphere_radius: Sphere radius.  Set to 0 to suppress spheres.
        animate:      Animate through frames (ignored when T == 1).
        interval:     Milliseconds between animation frames.
        width, height: Viewer dimensions in pixels.

    Returns:
        py3Dmol.view — call .show() to render inline.

    Example:
        ca = rigids_to_ca_trajectory(rigids, coord_scale)
        view = show_py3dmol_trajectory(ca, batch['aatype'], color='orange')
        view.show()
    """
    import py3Dmol

    seqres = _coerce_seqres(seqres)
    ca_arr = np.asarray(ca_traj, dtype=np.float32)
    if ca_arr.ndim == 2:
        ca_arr = ca_arr[None]           # [N, 3] → [1, N, 3]

    pdb_str = _build_pdb_string(ca_arr, seqres)
    _sc = sphere_color or _SPHERE_DEFAULTS.get(color, color)

    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(pdb_str, 'pdb')
    # stick uses the CONECT records to draw cylinders between consecutive CAs.
    # This is reliable for CA-only PDBs; cartoon requires N/CA/C atoms.
    view.setStyle({}, {'stick': {'radius': 0.15, 'color': color}})
    if sphere_radius > 0:
        view.addStyle({}, {'sphere': {'color': _sc, 'radius': sphere_radius}})
    view.zoomTo()
    if animate and ca_arr.shape[0] > 1:
        view.animate({'loop': 'forward', 'reps': 0, 'interval': interval})
    return view


def show_py3dmol_overlay(
    structures,
    seqres,
    *,
    colors=None,
    radius: float = 0.35,
    width: int = 800,
    height: int = 450,
):
    """Overlay multiple CA structures in a single py3Dmol view.

    Each structure is shown as a coloured Cα trace + sphere ("pearls on a
    string").  Useful for comparing a real MD frame (blue) against a generated
    conformation (orange).

    Only the first frame (index 0) of each structure is shown.  For animated
    comparison use show_py3dmol_trajectory on each structure separately.

    Args:
        structures: List of CA arrays, each [T, N, 3] or [N, 3] in Angstroms.
                    Only frame 0 is used from arrays with T > 1.
        seqres:     Shared sequence string (str) or aatype array/tensor [N].
                    All structures must be of the same protein.
        colors:     List of CSS colour strings, one per structure.
                    Defaults to ['blue', 'orange', 'green', 'red', ...].
        radius:     Trace / sphere radius in Ångströms.
        width, height: Viewer dimensions in pixels.

    Returns:
        py3Dmol.view — call .show() to render inline.

    Example:
        view = show_py3dmol_overlay(
            [ref_ca[0], gen_ca[0]], batch['aatype'],
            colors=['blue', 'orange'],
        )
        view.show()
    """
    import py3Dmol

    seqres = _coerce_seqres(seqres)
    if colors is None:
        colors = _DEFAULT_COLOR_CYCLE[:len(structures)]

    view = py3Dmol.view(width=width, height=height)
    for idx, (ca, color) in enumerate(zip(structures, colors)):
        ca_arr = np.asarray(ca, dtype=np.float32)
        if ca_arr.ndim == 2:
            ca_arr = ca_arr[None]       # [N, 3] → [1, N, 3]
        frame0 = ca_arr[[0]]            # always single frame for overlay
        pdb_str = _build_pdb_string(frame0, seqres)
        _sc = _SPHERE_DEFAULTS.get(color, color)
        view.addModel(pdb_str, 'pdb')
        view.setStyle({'model': idx}, {'stick': {'radius': 0.15, 'color': color}})
        view.addStyle({'model': idx}, {'sphere': {'color': _sc, 'radius': radius}})
    view.zoomTo()
    return view
