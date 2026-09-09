"""Minimal LoRA for the sparse-attention Linear layers of the TRELLIS.2 texture DiT."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

ATTN_LINEARS = ("to_qkv", "to_q", "to_kv", "to_out")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        delta = (self.dropout(x.float()) @ self.lora_A.t() @ self.lora_B.t()) * self.scale
        return y + delta.to(y.dtype)

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        w = self.base.weight.float() + (self.lora_B @ self.lora_A) * self.scale
        self.base.weight.copy_(w.to(self.base.weight.dtype))
        return self.base


def _attn_modules(flow_model, targets: set[str]):
    for block in flow_model.blocks:
        if "self" in targets and hasattr(block, "self_attn"):
            yield block.self_attn
        if "cross" in targets and hasattr(block, "cross_attn"):
            # v4 LegendCrossAttention wraps the original attention as .inner
            yield getattr(block.cross_attn, "inner", block.cross_attn)


def inject_lora(flow_model, r: int = 16, alpha: float = 32.0, targets=("self", "cross"), dropout: float = 0.0) -> list[nn.Parameter]:
    params = []
    for attn in _attn_modules(flow_model, set(targets)):
        for name in ATTN_LINEARS:
            lin = getattr(attn, name, None)
            if isinstance(lin, nn.Linear):
                wrapped = LoRALinear(lin, r, alpha, dropout)
                setattr(attn, name, wrapped)
                params += [wrapped.lora_A, wrapped.lora_B]
    return params


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def load_lora_state_dict(model: nn.Module, state: dict) -> None:
    missing = model.load_state_dict(state, strict=False)
    unexpected = [k for k in missing.unexpected_keys]
    if unexpected:
        raise KeyError(f"LoRA keys not in model: {unexpected[:5]}")
    loaded = {k for k in state}
    expected = {k for k in model.state_dict() if "lora_A" in k or "lora_B" in k}
    if loaded != expected:
        raise KeyError(f"LoRA key mismatch: {len(expected - loaded)} missing, {len(loaded - expected)} extra")


def merge_lora(flow_model) -> int:
    """Fold every LoRALinear back into a plain Linear. Returns the number of merged layers."""
    n = 0
    for attn in _attn_modules(flow_model, {"self", "cross"}):
        for name in ATTN_LINEARS:
            mod = getattr(attn, name, None)
            if isinstance(mod, LoRALinear):
                setattr(attn, name, mod.merged_linear())
                n += 1
    return n
