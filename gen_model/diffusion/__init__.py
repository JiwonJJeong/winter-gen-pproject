"""Shim: re-exports diffusion classes from extern/se3_diffusion (read-only submodule).

All diffusion logic lives in extern/se3_diffusion/data/.
These shims preserve existing import paths (gen_model.diffusion.*).
"""
from gen_model.diffusion.se3_diffuser import SE3Diffuser
from gen_model.diffusion.so3_diffuser import SO3Diffuser
from gen_model.diffusion.r3_diffuser import R3Diffuser

__all__ = ['SE3Diffuser', 'SO3Diffuser', 'R3Diffuser']
