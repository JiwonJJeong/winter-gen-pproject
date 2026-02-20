"""gen_model package for SE(3) diffusion on MD trajectory data."""

__version__ = '0.1.0'

# Backward-compat re-exports so that existing code using
# `from gen_model import residue_constants` etc. continues to work.
from gen_model.data import residue_constants   # noqa: F401
from gen_model.utils import rigid_utils        # noqa: F401
from gen_model.data import all_atom            # noqa: F401
