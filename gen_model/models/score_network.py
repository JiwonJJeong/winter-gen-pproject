"""Shim: re-exports ScoreNetwork from extern/se3_diffusion (read-only submodule).

All ScoreNetwork logic lives in extern/se3_diffusion/model/score_network.py.
This file exists only to preserve the import path gen_model.models.score_network.
"""
import gen_model.path_setup  # noqa: F401
from model.score_network import ScoreNetwork, Embedder  # noqa: F401

__all__ = ['ScoreNetwork', 'Embedder']
