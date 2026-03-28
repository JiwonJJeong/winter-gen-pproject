"""Shim: re-exports SO3Diffuser from extern/se3_diffusion (read-only submodule)."""
import gen_model.path_setup  # noqa: F401
from data.so3_diffuser import SO3Diffuser  # noqa: F401

__all__ = ['SO3Diffuser']
