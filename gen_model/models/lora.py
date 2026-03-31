"""Low-Rank Adaptation (LoRA) for the SE(3) diffusion ScoreNetwork.

Implements LoRA (Hu et al., 2021) as a drop-in wrapper for nn.Linear layers.
LoRA freezes the pre-trained weights and injects trainable low-rank matrices
A ∈ R^{r×d_in} and B ∈ R^{d_out×r} so that:

    h = W_0 x + (B A x) * (alpha / r)

At initialisation B=0, so the model starts identical to the frozen baseline.

Usage:
    from gen_model.models.lora import apply_lora
    model = ScoreNetwork(model_conf, diffuser)
    apply_lora(model, model_conf.lora)   # in-place; prints param count
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen nn.Linear base weight + trainable rank-r LoRA correction.

    Wraps any nn.Linear (including subclasses with custom init) and exposes
    .weight / .bias / .in_features / .out_features for downstream compatibility.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base  = base
        self.r     = r
        self.alpha = alpha

        d_out, d_in = base.weight.shape
        # A: random init scaled by 1/sqrt(r); B: zero init (ΔW=0 at start)
        self.lora_A = nn.Parameter(torch.randn(r, d_in) / r ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

        # Freeze base weights
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two sequential small matmuls (O(r·d)) instead of materialising the full
        # (d_out × d_in) product matrix B @ A at every forward call (O(d²)).
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return self.base(x) + lora_out * (self.alpha / self.r)

    # Expose base properties so code that inspects .weight/.bias still works
    @property
    def weight(self):       return self.base.weight
    @property
    def bias(self):         return self.base.bias
    @property
    def in_features(self):  return self.base.in_features
    @property
    def out_features(self): return self.base.out_features

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"r={self.r}, alpha={self.alpha}")


def inject_lora(module: nn.Module, target_names: set, r: int, alpha: float) -> None:
    """Recursively replace direct children whose name is in target_names with LoRALinear.

    Only replaces nn.Linear instances (catches custom Linear subclasses too).
    Modifies module in-place.
    """
    for name, child in list(module.named_children()):
        if name in target_names and isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, alpha))
        else:
            inject_lora(child, target_names, r, alpha)


def freeze_non_lora(module: nn.Module) -> None:
    """Freeze all parameters that are not part of a LoRA adapter (lora_A / lora_B)."""
    for name, p in module.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            p.requires_grad_(False)


def apply_lora(model: nn.Module, lora_conf) -> nn.Module:
    """Inject LoRA adapters and freeze non-LoRA weights.

    Args:
        model:      The ScoreNetwork (or any nn.Module) to modify in-place.
        lora_conf:  OmegaConf node with keys: enabled, r, alpha, target_modules.

    Returns:
        The same model object (modified in-place for convenience).
    """
    if not getattr(lora_conf, 'enabled', False):
        return model

    r            = int(lora_conf.r)
    alpha        = float(lora_conf.alpha)
    target_names = set(lora_conf.target_modules)

    inject_lora(model, target_names, r, alpha)
    freeze_non_lora(model)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LoRA] r={r}, alpha={alpha}, targets={sorted(target_names)}")
    print(f"[LoRA] trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model
