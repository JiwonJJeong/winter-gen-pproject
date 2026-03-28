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
