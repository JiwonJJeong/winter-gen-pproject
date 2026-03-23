"""2D Rotary Position Embedding for joint (residue, frame) tokens.

Each token in the spatio-temporal attention has a 2D position (seq_idx, frame_idx).
The head dimension is split in half:
  - First half  encodes residue (sequence) position via 1D RoPE
  - Second half encodes frame   (temporal)  position via 1D RoPE

Applied to Q and K before computing attention scores; V is left unchanged.

Reference: RoFormer (Su et al., 2021), extended to 2D following STAR-MD.
"""

import torch
import torch.nn as nn


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding.

    Args:
        head_dim: Dimension of each attention head. Must be divisible by 4
                  (split in half for seq/frame, each half uses rotate-half trick).
        base: Frequency base for the inverse frequencies. Default 10000.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, (
            f"head_dim must be divisible by 4 for 2D-RoPE, got {head_dim}"
        )
        half = head_dim // 2   # each 1D RoPE covers this many dims
        quarter = half // 2    # number of (cos, sin) frequency pairs per axis

        # inv_freq[d] = base^{-2d / half},  shape [quarter]
        inv_freq = 1.0 / (base ** (torch.arange(0, quarter).float() * 2.0 / half))
        self.register_buffer("inv_freq", inv_freq)
        self.head_dim = head_dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate by swapping and negating the two halves: [-x2, x1]."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def _build_cos_sin(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin tables for the given positions.

        Args:
            positions: [B, T] integer (or float) position indices.

        Returns:
            cos, sin: each [B, 1, T, half_dim] broadcastable over heads.
        """
        # [B, T, quarter]
        freqs = torch.einsum("bt, d -> btd", positions.float(), self.inv_freq)
        # Duplicate so rotate-half works on the full half_dim: [B, T, half]
        emb = torch.cat([freqs, freqs], dim=-1)
        # Unsqueeze head dim for broadcasting: [B, 1, T, half]
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        seq_idx: torch.Tensor,
        frame_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply 2D-RoPE to a Q or K tensor.

        Args:
            x:          [B, H, L*N, head_dim]  query or key tensor.
            seq_idx:    [B, N]   residue position indices (integer).
            frame_idx:  [L]      frame position indices (integer).

        Returns:
            Tensor of the same shape as x with 2D-RoPE applied.

        Token layout:
            Token t = ℓ*N + i  →  residue i in frame ℓ.
            seq_pos[t]   = seq_idx[i]    (varies over N)
            frame_pos[t] = frame_idx[ℓ]  (varies over L)
        """
        B, H, T, D = x.shape
        L = frame_idx.shape[0]
        N = seq_idx.shape[1]
        assert T == L * N, (
            f"Token count mismatch: x has T={T} but L*N={L}*{N}={L*N}"
        )

        # ---- Build flat position tensors [B, T] -------------------------
        # seq_pos[b, ℓ*N + i] = seq_idx[b, i]
        seq_pos = (
            seq_idx.unsqueeze(1)          # [B, 1, N]
            .expand(B, L, N)              # [B, L, N]
            .reshape(B, T)                # [B, T]
        )
        # frame_pos[b, ℓ*N + i] = frame_idx[ℓ]
        frame_pos = (
            frame_idx.unsqueeze(-1)       # [L, 1]
            .expand(L, N)                 # [L, N]
            .reshape(T)                   # [T]
            .unsqueeze(0)                 # [1, T]
            .expand(B, T)                 # [B, T]
        )

        # ---- Build cos/sin tables [B, 1, T, half] -----------------------
        cos_seq,   sin_seq   = self._build_cos_sin(seq_pos)
        cos_frame, sin_frame = self._build_cos_sin(frame_pos)

        # ---- Rotate each half of the head dimension ---------------------
        half = D // 2
        x_seq   = x[..., :half]   # [B, H, T, half]
        x_frame = x[..., half:]   # [B, H, T, half]

        x_seq   = x_seq   * cos_seq   + self._rotate_half(x_seq)   * sin_seq
        x_frame = x_frame * cos_frame + self._rotate_half(x_frame) * sin_frame

        return torch.cat([x_seq, x_frame], dim=-1)
