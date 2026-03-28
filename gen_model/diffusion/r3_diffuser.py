"""Shim: re-exports R3Diffuser from extern/se3_diffusion (read-only submodule)."""
import gen_model.path_setup  # noqa: F401
from data.r3_diffuser import R3Diffuser  # noqa: F401

__all__ = ['R3Diffuser']
