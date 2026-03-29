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
  - KV-cache support for efficient autoregressive inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import gen_model.path_setup  # noqa: F401
from model.ipa_pytorch import Linear
from gen_model.models.rope2d import RoPE2D
from gen_model.models.adaln import AdaLN


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
        spatial_sigma: float = 0.0,
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
        self.spatial_sigma = spatial_sigma

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
        L_q: int,
        N: int,
        mask_q: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        L_kv: int = 0,
        mask_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build additive attention bias combining causal and padding masks.

        Args:
            L_q:     Number of query frames (new frames).
            L_kv:    Total number of key/value frames (cached + new).
                     When 0, defaults to L_q (no cache).
            mask_q:  [B, L_q, N] mask for query frames.
            mask_kv: [B, L_kv, N] mask for all KV frames. If None, uses mask_q.

        Returns:
            attn_bias: [B, 1, L_q*N, L_kv*N]
        """
        if L_kv == 0:
            L_kv = L_q
        T_q = L_q * N
        T_kv = L_kv * N

        if mask_kv is None:
            mask_kv = mask_q

        attn_bias = torch.zeros(B, 1, T_q, T_kv, device=device, dtype=dtype)

        # Block-causal mask: query at frame ℓq cannot attend to frame ℓkv > ℓq.
        # Query frames are the LAST L_q frames in the KV sequence.
        if self.causal and L_kv > 1:
            L_cached = L_kv - L_q
            q_frame = torch.arange(L_cached, L_kv, device=device)     # [L_q]
            kv_frame = torch.arange(L_kv, device=device)              # [L_kv]
            future = kv_frame[None, :] > q_frame[:, None]             # [L_q, L_kv]
            future_token = future.repeat_interleave(N, dim=0).repeat_interleave(N, dim=1)
            attn_bias = attn_bias.masked_fill(future_token[None, None], float("-inf"))

        # Padding mask on KV side
        pad_kv = (mask_kv == 0).reshape(B, T_kv)
        attn_bias = attn_bias.masked_fill(pad_kv[:, None, None, :], float("-inf"))

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
        kv_cache: Optional[dict] = None,
        ca_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            s_frames:  [B, L, N, c_s]  node embeddings for L (new) frames.
            frame_idx: [L_new]          frame position indices for the new frames.
            seq_idx:   [B, N]           integer residue position indices for RoPE.
            mask:      [B, L, N]        1 = valid residue, 0 = padded (new frames).
            cond:      [B, cond_dim]    AdaLN conditioning (t_emb || delta_t_emb).
            kv_cache:  Optional dict with 'k', 'v' [B, H, T_cached, D] from
                       previously finalized frames (read-only, not modified).
            ca_pos:    Optional [B, L, N, 3] CA positions for spatial Gaussian bias.
                       When spatial_sigma > 0, adds a soft distance-based falloff
                       so distant residues attend less strongly (SinFusion-inspired
                       local receptive field to prevent overfitting on single
                       trajectories). Ignored when spatial_sigma == 0.

        Returns:
            (output, new_kv):
              output: [B, L, N, c_s]  residual delta for the new frames.
              new_kv: dict with 'k', 'v' [B, H, L*N, D] for the new frames
                      (RoPE-applied K and raw V). Caller decides whether to
                      append to the persistent cache.
        """
        B, L, N, _ = s_frames.shape
        T_new = L * N
        H, D = self.num_heads, self.head_dim

        # 1. AdaLN input normalisation conditioned on t and delta_t
        x = self.adaln(s_frames, cond)               # [B, L, N, c_s]

        # 2. Flatten frames → tokens
        x_flat = x.reshape(B, T_new, self.c_s)       # [B, T_new, c_s]

        # 3. Project to Q, K, V
        q = self.q_proj(x_flat)                       # [B, T_new, c_s]
        k_new = self.k_proj(x_flat)
        v_new = self.v_proj(x_flat)

        # Reshape to multi-head: [B, H, T, D]
        def split_heads(t):
            return t.reshape(B, -1, H, D).permute(0, 2, 1, 3)

        q = split_heads(q)          # [B, H, T_new, D]
        k_new = split_heads(k_new)  # [B, H, T_new, D]
        v_new = split_heads(v_new)  # [B, H, T_new, D]

        # 4. Apply 2D-RoPE to Q and new K
        q = self.rope(q, seq_idx=seq_idx, frame_idx=frame_idx)
        k_new = self.rope(k_new, seq_idx=seq_idx, frame_idx=frame_idx)

        # Store new K/V for potential caching by caller
        new_kv = {'k': k_new.detach(), 'v': v_new.detach()}

        # 5. Prepend cached K/V if available
        L_cached = 0
        if kv_cache is not None and 'k' in kv_cache:
            L_cached = kv_cache['k'].shape[2] // N
            k = torch.cat([kv_cache['k'], k_new], dim=2)  # [B, H, T_total, D]
            v = torch.cat([kv_cache['v'], v_new], dim=2)
        else:
            k = k_new
            v = v_new

        L_total = L_cached + L

        # 6. Build attention mask accounting for cached frames
        if L_cached > 0:
            cached_mask = torch.ones(B, L_cached, N, device=s_frames.device, dtype=mask.dtype)
            full_mask_kv = torch.cat([cached_mask, mask], dim=1)  # [B, L_total, N]
        else:
            full_mask_kv = mask

        attn_bias = self._build_attn_bias(
            B, L, N, mask, s_frames.device, q.dtype,
            L_kv=L_total, mask_kv=full_mask_kv)

        # 6b. Spatial Gaussian bias (SinFusion-inspired local receptive field)
        #     Adds -(d_ij / sigma)^2 to attention logits so distant residues
        #     contribute exponentially less.  Only the spatial (within-frame)
        #     component is penalised; cross-frame pairs of the same residue
        #     are unaffected (distance = 0 within a token's own residue).
        if self.spatial_sigma > 0 and ca_pos is not None:
            # ca_pos: [B, L, N, 3] → flatten to [B, T_new, 3]
            ca_flat = ca_pos.reshape(B, T_new, 3)
            if L_cached > 0 and kv_cache is not None:
                # For cached context, we don't have CA positions — assume
                # same-residue tokens across frames have zero spatial penalty.
                # Build a dummy spatial bias of zeros for cached tokens.
                spatial_bias = torch.zeros(B, 1, T_new, L_total * N,
                                           device=s_frames.device, dtype=q.dtype)
                # Only compute for the new×new block
                new_dist = torch.cdist(ca_flat, ca_flat)  # [B, T_new, T_new]
                spatial_bias[:, :, :, L_cached * N:] = -(new_dist / self.spatial_sigma).pow(2).unsqueeze(1)
            else:
                dist = torch.cdist(ca_flat, ca_flat)  # [B, T, T]
                spatial_bias = -(dist / self.spatial_sigma).pow(2).unsqueeze(1)  # [B, 1, T, T]
            attn_bias = attn_bias + spatial_bias

        # 7. Scaled dot-product attention
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + attn_bias
        attn_weights = torch.softmax(attn_logits, dim=-1)

        out = torch.matmul(attn_weights, v)           # [B, H, T_new, D]

        # 8. Merge heads and unflatten back to [B, L, N, c_s]
        out = out.permute(0, 2, 1, 3).reshape(B, L, N, self.c_s)

        # 9. Output projection (zero-init → identity residual at init)
        out = self.out_proj(out)

        return out, new_kv
