"""Flat MLP baseline (SPEC §13): 80 -> 128 -> 64 -> 5 with ReLU, plus an optional value head.

Input is the ``[prev, cur]`` stack of ``drone_env.wrappers.flatten_obs`` frames (2 x 40). The
2-frame stack is input, not memory. The MLP never sees the action mask; callers apply it.
"""
from __future__ import annotations

import torch.nn as nn


class FlatMLP(nn.Module):
    def __init__(self, in_dim: int = 80, hidden=(128, 64), value_head: bool = True):
        super().__init__()
        self.cfg = {"in_dim": in_dim, "hidden": list(hidden), "value_head": value_head}
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.pi = nn.Linear(d, 5)
        self.value = nn.Linear(d, 1) if value_head else None

    def forward(self, obs):
        """Returns (logits (B, 5) unmasked, value (B,) or None)."""
        z = self.body(obs)
        return self.pi(z), (self.value(z).squeeze(-1) if self.value is not None else None)


def mlp_from_config(cfg: dict) -> FlatMLP:
    return FlatMLP(in_dim=int(cfg.get("in_dim", 80)), hidden=tuple(cfg.get("hidden", (128, 64))),
                   value_head=bool(cfg.get("value_head", True)))
