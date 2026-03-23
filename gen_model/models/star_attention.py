"""Spatio-Temporal Attention for STAR-MD.

Implements joint (N×L)-token multi-head self-attention over L MD frames,
following "Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein
Dynamics".

Token layout: token t = ℓ*N + i  (residue i in frame ℓ).

Key properties:
  - Block-causal mask: frame ℓ attends only to frames ℓ' ≤ ℓ (sorted by frame_idx).
  - 2D-RoPE on Q and K encoding (residue position, frame position) jointly.
  - AdaLN input conditioning on diffusion time t and physical stride Δt.
  - Output projection zero-initialized ("final" init) so the residual is identity
    at the start of training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gen_model.models.ipa_pytorch import Linear
from gen_model.models.rope2d import RoPE2D
from gen_model.models.modules import AdaLN


class SpatioTemporalAttention(nn.Module):
    """Joint spatio-temporal multi-head self-attention over L frames.

    Args:
        c_s:       Node embedding dimension (input and output channel count).
        num_heads: Number of attention heads. c_s must be divisible by num_heads,
                   and head_dim (= c_s // num_heads) must be divisible by 4 for
                   2D-RoPE (split in half per axis, rotate-half trick).
        cond_dim:  Dimension of the AdaLN conditioning vector (default 64 = 2*32).
        causal:    If True (default), apply block-causal mask so frame ℓ cannot
                   attend to frame ℓ' > ℓ. Set False only for ablations.
    """

    def __init__(
        self,
        c_s: int,
        num_heads: int,
        cond_dim: int = 64,
        causal: bool = True,
    ):
        super().__init__()
        assert c_s % num_heads == 0, (
            f"c_s={c_s} must be divisible by num_heads={num_heads}"
        )
        head_dim = c_s // num_heads
        assert head_dim % 4 == 0, (
            f"head_dim={head_dim} must be divisible by 4 for 2D-RoPE"
        )

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.causal = causal

        # Input conditioning
        self.adaln = AdaLN(c_s=c_s, cond_dim=cond_dim)

        # QKV projections — LeCun default init
        self.q_proj = Linear(c_s, c_s, init="default")
        self.k_proj = Linear(c_s, c_s, init="default")
        self.v_proj = Linear(c_s, c_s, init="default")

        # Output projection — zero-init so the residual starts as identity
        self.out_proj = Linear(c_s, c_s, init="final")

        # 2D Rotary Position Embedding
        self.rope = RoPE2D(head_dim=head_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_attn_bias(
        self,
        B: int,
        L: int,
        N: int,
        mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build additive attention bias combining causal and padding masks.

        Returns:
            attn_bias: [B, 1, L*N, L*N]  (head dim broadcast)
        """
        T = L * N
        attn_bias = torch.zeros(B, 1, T, T, device=device, dtype=dtype)

        # Block-causal mask: token at frame ℓ1 cannot attend to frame ℓ2 > ℓ1.
        # Build a [L, L] block mask then expand to [L*N, L*N].
        if self.causal and L > 1:
            # future_mask[ℓ1, ℓ2] = True  iff ℓ2 > ℓ1
            future = torch.triu(
                torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
            )  # [L, L]
            # Expand to token level: each frame block is N×N
            # [L, L] -> [L*N, L*N] via repeat_interleave
            future_token = future.repeat_interleave(N, dim=0).repeat_interleave(N, dim=1)  # [L*N, L*N]
            attn_bias = attn_bias.masked_fill(future_token[None, None], float("-inf"))

        # Padding mask: zero out attention *to* any padded token.
        # mask: [B, L, N]  (1 = valid, 0 = padded)
        # A padded token as key should receive -inf so it never contributes.
        pad_token = (mask == 0).reshape(B, T)  # [B, T]  True = padded
        # Broadcast over query dim: [B, 1, 1, T]
        attn_bias = attn_bias.masked_fill(pad_token[:, None, None, :], float("-inf"))

        return attn_bias

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        s_frames: torch.Tensor,
        frame_idx: torch.Tensor,
        seq_idx: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s_frames:  [B, L, N, c_s]  node embeddings for L frames.
            frame_idx: [L]              integer frame position indices for RoPE.
            seq_idx:   [B, N]           integer residue position indices for RoPE.
            mask:      [B, L, N]        1 = valid residue, 0 = padded.
            cond:      [B, cond_dim]    AdaLN conditioning (t_emb || delta_t_emb).

        Returns:
            [B, L, N, c_s]  updated node embeddings (residual connection applied
                            outside this module by the caller in IpaScore).
        """
        B, L, N, _ = s_frames.shape
        T = L * N
        H, D = self.num_heads, self.head_dim

        # 1. AdaLN input normalisation conditioned on t and delta_t
        x = self.adaln(s_frames, cond)               # [B, L, N, c_s]

        # 2. Flatten frames → tokens
        x_flat = x.reshape(B, T, self.c_s)           # [B, T, c_s]

        # 3. Project to Q, K, V
        q = self.q_proj(x_flat)                      # [B, T, c_s]
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        # Reshape to multi-head: [B, H, T, D]
        def split_heads(t):
            return t.reshape(B, T, H, D).permute(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # 4. Apply 2D-RoPE to Q and K
        q = self.rope(q, seq_idx=seq_idx, frame_idx=frame_idx)
        k = self.rope(k, seq_idx=seq_idx, frame_idx=frame_idx)

        # 5-6. Scaled dot-product attention with causal + padding bias
        attn_bias = self._build_attn_bias(B, L, N, mask, s_frames.device, q.dtype)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        attn_logits = attn_logits + attn_bias
        attn_weights = torch.softmax(attn_logits, dim=-1)

        out = torch.matmul(attn_weights, v)          # [B, H, T, D]

        # 7. Merge heads and unflatten back to [B, L, N, c_s]
        out = out.permute(0, 2, 1, 3).reshape(B, L, N, self.c_s)  # [B, L, N, c_s]

        # 8. Output projection (zero-init → identity residual at init)
        out = self.out_proj(out)                     # [B, L, N, c_s]

        return out
