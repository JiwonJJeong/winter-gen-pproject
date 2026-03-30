"""Add extern submodules to sys.path so their internal imports resolve correctly.

Import this module (or call setup()) before importing anything from
extern/se3_diffusion, extern/mdgen, or extern/sinfusion.

Usage:
    import gen_model.path_setup  # side-effect: extends sys.path

Each submodule is inserted once — safe to import multiple times.
"""
import sys
from pathlib import Path

_EXTERN = Path(__file__).parents[1] / "extern"

_PATHS = [
    _EXTERN / "se3_diffusion",   # exposes: model/, data/, openfold/
    _EXTERN / "mdgen",           # exposes: mdgen/
    _EXTERN / "sinfusion",       # exposes: models/, diffusion/, datasets/
]

for _p in _PATHS:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


# ---------------------------------------------------------------------------
# Monkey-patch openfold's rot_to_quat to fix float64 promotion from eigh.
# torch.linalg.eigh promotes float32 → float64 on CPU, which causes dtype
# mismatches during backward.  The patch casts the output back to the input
# dtype.  Applied here so every code path (including upstream IpaScore) is
# covered without modifying the extern submodule.
# ---------------------------------------------------------------------------
def _patch_rot_to_quat():
    import openfold.utils.rigid_utils as _of_ru
    _orig = _of_ru.rot_to_quat

    def _fixed_rot_to_quat(rot):
        import torch
        result = _orig(rot)
        # k inherits dtype from rot entries; eigh may promote it
        return result.to(rot.dtype if isinstance(rot, torch.Tensor) else result.dtype)

    _of_ru.rot_to_quat = _fixed_rot_to_quat

_patch_rot_to_quat()
