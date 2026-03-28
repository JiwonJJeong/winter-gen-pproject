"""Shim: re-exports SE3Diffuser from extern/se3_diffusion (read-only submodule)."""
import gen_model.path_setup  # noqa: F401
from data.se3_diffuser import SE3Diffuser  # noqa: F401

__all__ = ['SE3Diffuser']
