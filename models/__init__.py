"""Models: Arc (System 1) and the flat MLP baseline, plus checkpoint I/O.

Checkpoint (``runs/<run>/best.pt``): a dict with ``arch`` ("arc" | "mlp"), ``model_cfg``,
``state_dict``, ``stage`` ("supervised" | "dagger" | "ppo"), ``run``, ``epoch`` and the
training ``config``.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .mlp import FlatMLP, mlp_from_config
from .arc import ArcNet, arc_from_config

ARCHES = ("arc", "mlp")


def build_model(arch: str, model_cfg: dict | None = None) -> torch.nn.Module:
    model_cfg = model_cfg or {}
    if arch == "arc":
        return arc_from_config(model_cfg)
    if arch == "mlp":
        return mlp_from_config(model_cfg)
    raise ValueError(f"unknown arch {arch!r}; choose one of {ARCHES}")


def save_checkpoint(path, model, arch: str, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = {"arch": arch, "model_cfg": dict(model.cfg), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    ck.update(extra)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(path)


def load_checkpoint(path, map_location="cpu"):
    """-> (model in eval mode, checkpoint dict)."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ck["arch"], ck["model_cfg"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def policy_outputs(model, arch: str, batch: dict) -> dict:
    """Uniform forward for both archs. ``batch`` holds Arc inputs (self, rays, hazards,
    hazard_type, hazard_mask, act_avail [, act_desc]) or ``obs`` (B, 80) for the MLP, plus
    ``act_avail``. Returns masked ``logits`` and, where the arch has them, ``safe_logits``,
    ``threat_logits``, ``decide_attn`` and ``value``."""
    avail = batch["act_avail"]
    if arch == "mlp":
        logits, value = model(batch["obs"])
        return {"logits": logits.masked_fill(~avail, float("-inf")), "value": value}
    B = avail.shape[0]
    act_ids = batch.get("act_ids")
    if act_ids is None:
        act_ids = torch.arange(5, device=avail.device).expand(B, 5)
    act_desc = batch.get("act_desc")
    if act_desc is None:
        from drone_env.env import action_descriptors
        act_desc = torch.from_numpy(action_descriptors()).to(avail.device).expand(B, 5, 3)
    lg, sf, th, attn, v = model(batch["self"], batch["rays"], batch["hazards"], batch["hazard_type"],
                                batch["hazard_mask"], act_ids, act_desc, avail)
    return {"logits": lg, "safe_logits": sf, "threat_logits": th, "decide_attn": attn, "value": v}
