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
        # eigh may promote float32 → float64; cast back to input dtype
        return result.to(rot.dtype if isinstance(rot, torch.Tensor) else result.dtype)

    _of_ru.rot_to_quat = _fixed_rot_to_quat

_patch_rot_to_quat()


# ---------------------------------------------------------------------------
# Monkey-patch SO3Diffuser.torch_score to fix Float/Double mismatch.
#
# torch_score creates sigma via torch.tensor(numpy_float64) — always Double.
# omega is derived from the model (Float).  When igso3_expansion mixes them,
# the autograd graph records Double ops but the loss gradient is Float, causing:
#   RuntimeError: Found dtype Float but expected Double
#
# Fix: run torch_score in Double (matching sigma), cast result back to Float.
# The .double()/.to() boundary is transparent to autograd.
# ---------------------------------------------------------------------------
def _patch_so3_torch_score():
    import torch
    from data import so3_diffuser as _so3

    _orig = _so3.SO3Diffuser.torch_score

    def _fixed_torch_score(self, vec, t, eps=1e-6):
        result = _orig(self, vec.double(), t, eps)
        return result.to(vec.dtype)

    _so3.SO3Diffuser.torch_score = _fixed_torch_score

_patch_so3_torch_score()
