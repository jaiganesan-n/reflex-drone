"""Fly a PyTorch checkpoint in the real env (training-time evaluation, DAgger, calibration).

* ``obs_batch`` turns observation dicts into model inputs (SPEC §5 -> §13 casts).
* ``TorchPolicy``: one env, one frame at a time (worker processes).
* ``fly_lockstep``: many seeds at once, one batched forward per frame (online evaluation).

The app and the final evaluation use the ONNX agents in ``agents/`` instead.
"""
from __future__ import annotations

import numpy as np
import torch

from drone_env.wrappers import FlatHistory


def obs_batch(obs_list, act_desc: np.ndarray, arch: str, flat80: np.ndarray | None = None) -> dict:
    avail = torch.from_numpy(np.stack([o["action_mask"] for o in obs_list]).astype(bool))
    if arch == "mlp":
        return {"obs": torch.from_numpy(np.asarray(flat80, np.float32)), "act_avail": avail}
    B = len(obs_list)
    return {
        "self": torch.from_numpy(np.stack([o["self"] for o in obs_list]).astype(np.float32)),
        "rays": torch.from_numpy(np.stack([o["rays"] for o in obs_list]).astype(np.float32)),
        "hazards": torch.from_numpy(np.stack([o["hazards"] for o in obs_list]).astype(np.float32)),
        "hazard_type": torch.from_numpy(np.stack([o["hazard_type"] for o in obs_list]).astype(np.int64)),
        "hazard_mask": torch.from_numpy(np.stack([o["hazard_mask"] for o in obs_list]).astype(bool)),
        "act_avail": avail,
        "act_ids": torch.arange(5).expand(B, 5),
        "act_desc": torch.from_numpy(np.asarray(act_desc, np.float32)).expand(B, 5, 3),
    }


def masked_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """softmax(logits / T) with -inf (unavailable) entries -> 0."""
    return torch.softmax(logits / temperature, dim=-1)


class TorchPolicy:
    """Single-env policy for worker processes: ``act(obs) -> (action, probs)``."""

    def __init__(self, ckpt_path, temperature: float = 1.0, model=None, arch: str | None = None):
        from models import load_checkpoint
        torch.set_num_threads(1)
        if model is None:
            model, ck = load_checkpoint(ckpt_path)
            arch = ck["arch"]
        self.model, self.arch = model.eval(), arch
        self.T = float(temperature)
        self.hist = FlatHistory() if arch == "mlp" else None
        self.act_desc = None

    def reset(self, action_desc: np.ndarray) -> None:
        self.act_desc = np.asarray(action_desc, np.float32)
        if self.hist is not None:
            self.hist.reset()

    @torch.inference_mode()
    def act(self, obs: dict):
        from models import policy_outputs
        flat = self.hist.push(obs)[None] if self.hist is not None else None
        out = policy_outputs(self.model, self.arch, obs_batch([obs], self.act_desc, self.arch, flat))
        self.last_logits = out["logits"][0].numpy()
        p = masked_probs(out["logits"][0], self.T).numpy()
        return int(np.argmax(p)), p


@torch.inference_mode()
def fly_lockstep(model, arch: str, level: str, seeds, frame_cap: int) -> list[dict]:
    """Fly the greedy policy on every seed in lockstep (one batched forward per frame)."""
    from drone_env.env import DroneDashEnv
    from models import policy_outputs
    model.eval()
    envs, obs, hists, alive, res = [], [], [], [], {}
    for s in seeds:
        e = DroneDashEnv(level=level, render_mode=None, max_frames=frame_cap)
        o, _ = e.reset(seed=int(s))
        envs.append(e)
        obs.append(o)
        if arch == "mlp":
            h = FlatHistory()
            h.reset()
            hists.append(h)
        alive.append(True)
    act_desc = envs[0].action_descriptors()
    stacks = [None] * len(envs)
    while any(alive):
        idx = [i for i, a in enumerate(alive) if a]
        flat = None
        if arch == "mlp":
            flat = np.stack([hists[i].push(obs[i]) for i in idx])
        out = policy_outputs(model, arch, obs_batch([obs[i] for i in idx], act_desc, arch, flat))
        acts = out["logits"].argmax(-1).tolist()
        for j, i in enumerate(idx):
            o, _r, term, trunc, info = envs[i].step(int(acts[j]))
            obs[i] = o
            if term or trunc:
                alive[i] = False
                res[i] = {"level": level, "seed": int(seeds[i]), "distance_m": float(info["distance_m"]),
                          "frames": int(info["frame"]), "crash_cause": info["crash_cause"]}
    return [res[i] for i in range(len(envs))]
