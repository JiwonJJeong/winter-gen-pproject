"""Adaptive Layer Normalization for STAR-MD diffusion conditioning.

AdaLN(x, cond) = γ(cond) ⊙ LayerNorm(x) + β(cond)

γ and β are each a two-layer MLP: Linear(cond_dim, c_s) → SiLU → Linear(c_s, c_s).
The final linear in each MLP is zero-initialized so that γ=1, β=0 at the start of
training — the block is an identity residual until the network learns to use it.

Used inside SpatioTemporalAttention to condition cross-frame attention on both the
diffusion timestep t and the physical stride delta_t.
"""

import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on an external vector.

    Args:
        c_s:      Feature dimension of the input x (the dimension being normalised).
        cond_dim: Dimension of the conditioning vector (typically 2*D = 64, where
                  D = index_embed_size = 32, carrying [t_emb, delta_t_emb]).
    """

    def __init__(self, c_s: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(c_s, elementwise_affine=False)

        # γ MLP: produces per-channel scale; initialised so γ = 1
        self.gamma_mlp = nn.Sequential(
            nn.Linear(cond_dim, c_s),
            nn.SiLU(),
            nn.Linear(c_s, c_s),
        )
        # β MLP: produces per-channel bias; initialised so β = 0
        self.beta_mlp = nn.Sequential(
            nn.Linear(cond_dim, c_s),
            nn.SiLU(),
            nn.Linear(c_s, c_s),
        )

        # Zero-init the final linear in each MLP so the block starts as identity.
        nn.init.zeros_(self.gamma_mlp[-1].weight)
        nn.init.zeros_(self.gamma_mlp[-1].bias)   # γ offset = 0 → effective γ = 1 + 0 = 1
        nn.init.zeros_(self.beta_mlp[-1].weight)
        nn.init.zeros_(self.beta_mlp[-1].bias)   # β = 0

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [..., c_s]                 input features (any leading dims).
            cond: [B, cond_dim]              per-batch conditioning, OR
                  [B, L, cond_dim]           per-(batch, frame) conditioning
                                             (Diffusion Forcing).

        Returns:
            Tensor of the same shape as x.
        """
        gamma = self.gamma_mlp(cond)
        beta  = self.beta_mlp(cond)
        # Broadcast cond's leading dims to match x by inserting singleton axes
        # *before* the channel dim (not at axis 1). For x=[B, L, N, c_s]:
        #   - cond=[B, c_s]    → gamma=[B, 1, 1, c_s]
        #   - cond=[B, L, c_s] → gamma=[B, L, 1, c_s]   (per-frame cond)
        # Inserting at axis 1 would misalign cond's L dim with x's N dim.
        for _ in range(x.dim() - cond.dim()):
            gamma = gamma.unsqueeze(-2)
            beta  = beta.unsqueeze(-2)

        # (1 + gamma) keeps the structural 1 outside the MLP so the scale can
        # never flip sign regardless of how far gamma drifts during training.
        return (1 + gamma) * self.norm(x) + beta
