"""Minimal utilities for gen_model."""
import torch
import numpy as np


def calc_distogram(pos, min_bin, max_bin, num_bins):
    """Calculate distance histogram/distogram from positions.
    
    Args:
        pos: [..., N, 3] positions
        min_bin: minimum distance bin
        max_bin: maximum distance bin
        num_bins: number of bins
        
    Returns:
        [..., N, N, num_bins] distogram
    """
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram
