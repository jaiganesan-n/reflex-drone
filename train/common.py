"""Shared training utilities: configs, devices, run folders, the contract loss, offline and
online evaluation, and the end-of-run VAL ratio that the gates read."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from drone_env import constants as C
from drone_env.seeds import seeds_for

ROOT = Path(__file__).resolve().parents[1]

SUPERVISED_DEFAULTS = {
    "arch": "arc", "model": {}, "data": "data/teacher", "stage": "supervised",
    "epochs": 10, "batch_size": 512, "lr": 3.0e-4, "weight_decay": 0.01, "schedule": "cosine",
    "warmup_frac": 0.05, "grad_clip": 1.0, "loss": {"kl": 1.0, "safe": 0.5, "threat": 0.25},
    "device": "auto", "threads": 0, "seed": 0,
    "gate_level": "L3", "online_eval": True, "online_seeds": 10, "eval_frame_cap": 6000,
    "final_val_seeds": 50, "planner_cache_dir": "reports", "workers": 6,
    "holdout_max_frames": 20000, "overfit_frames": 0, "steps_per_epoch": 0, "assert_loss_below": None,
    "tau": None, "init": None, "burst_weight": 1.0,
}


# ------------------------------------------------------------------------------------------
# configs and run folders
# ------------------------------------------------------------------------------------------
def load_config(path, defaults: dict) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    cfg = json.loads(json.dumps(defaults))
    for k, v in raw.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def as_list(x) -> list[str]:
    if x is None:
        return []
    return [str(v) for v in x] if isinstance(x, (list, tuple)) else [str(x)]


class Run:
    """``runs/<name>/`` with config.yaml, metrics.jsonl, best.pt, last.pt and summary.json."""

    def __init__(self, name: str, cfg: dict, root: Path | None = None, fresh_metrics: bool = True):
        self.name = name
        self.dir = Path(root or "runs") / name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        self.metrics_path = self.dir / "metrics.jsonl"
        if fresh_metrics and self.metrics_path.exists():
            self.metrics_path.write_text("")

    def log(self, row: dict) -> None:
        row = {"time": round(time.time(), 1), **row}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def path(self, name: str) -> Path:
        return self.dir / name

    def write_summary(self, summary: dict) -> None:
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))


def set_threads(n: int) -> None:
    torch.set_num_threads(n if n and n > 0 else max(1, os.cpu_count() or 1))


def select_device(spec: str, bench_fn=None) -> torch.device:
    """``cpu``, ``mps`` or ``auto``: MPS only if available and measurably faster in a quick
    benchmark of ``bench_fn(device) -> seconds``; otherwise CPU."""
    if spec == "cpu":
        return torch.device("cpu")
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if spec == "mps":
        if not mps_ok:
            print("[device] MPS not available; using CPU")
            return torch.device("cpu")
        return torch.device("mps")
    if not mps_ok or bench_fn is None:
        return torch.device("cpu")
    try:
        t_cpu = bench_fn(torch.device("cpu"))
        t_mps = bench_fn(torch.device("mps"))
    except Exception as e:                      # pragma: no cover - hardware specific
        print(f"[device] MPS benchmark failed ({e}); using CPU")
        return torch.device("cpu")
    dev = torch.device("mps") if t_mps < 0.9 * t_cpu else torch.device("cpu")
    print(f"[device] auto: cpu {t_cpu * 1000:.0f} ms vs mps {t_mps * 1000:.0f} ms per step -> {dev}")
    return dev


def lr_lambda(total_steps: int, warmup_frac: float, schedule: str):
    warm = int(round(total_steps * warmup_frac))

    def f(step: int) -> float:
        if schedule == "flat":
            return 1.0
        if warm > 0 and step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f


# ------------------------------------------------------------------------------------------
# loss
# ------------------------------------------------------------------------------------------
def contract_loss(out: dict, batch: dict, arch: str, weights: dict | None = None) -> tuple[torch.Tensor, dict]:
    """L = KL(p_teacher || p) + 0.5 * sum_a BCE(safe_a) + 0.25 * CE(threat) (SPEC §13).

    Masked actions are excluded from the KL and BCE terms (they get no gradient). The MLP uses
    the KL term only. ``out["logits"]`` may be raw or already masked. Optional ``hard`` weight (0 by
    default): + hard * CE(teacher's chosen action), for both archs. With the soft labels at T = 2 a
    choice the teacher wins by a few hundredths (start climbing now, or hold) is nearly uniform."""
    w = {"kl": 1.0, "safe": 0.5, "threat": 0.25, **(weights or {})}
    avail = batch["act_avail"]
    logits = out["logits"].masked_fill(~avail, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1).masked_fill(~avail, 0.0)
    pt = batch["p_teacher"] * avail
    logpt = torch.log(pt.clamp_min(1e-12))
    kl = torch.where(pt > 0, pt * (logpt - logp), torch.zeros_like(pt)).sum(-1).mean()
    parts = {"kl": kl}
    loss = w["kl"] * kl
    if w.get("hard"):
        hard = -logp.gather(-1, batch["action"][:, None]).mean()
        parts["hard"] = hard
        loss = loss + w["hard"] * hard
    if arch != "mlp":
        av = avail.float()
        bce = F.binary_cross_entropy_with_logits(out["safe_logits"].masked_fill(~avail, 0.0), batch["safe"],
                                                 reduction="none")
        bce = (bce * av).sum(-1).mean()
        ce = F.cross_entropy(out["threat_logits"], batch["threat"])
        parts.update(bce=bce, ce=ce)
        loss = loss + w["safe"] * bce + w["threat"] * ce
    parts["loss"] = loss
    return loss, parts


# ------------------------------------------------------------------------------------------
# offline metrics on holdout frames
# ------------------------------------------------------------------------------------------
def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC by the rank-sum formula (ties get average ranks). NaN if one class is empty."""
    labels = labels.astype(bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def top_set_agreement(logits: torch.Tensor, p_teacher: torch.Tensor, tie_tol: float = 0.1,
                      label_temperature: float = 2.0) -> torch.Tensor:
    """True where the model's argmax is one of the teacher's best actions. Teacher scores tie
    often (e.g. HOLD, DESCEND and DUCK are equally good in open air), so comparing with the
    teacher's lowest-index argmax would undercount agreement. Scores within ``tie_tol`` of the
    best count as tied (in p_teacher space: p >= max_p * exp(-tie_tol / T))."""
    pick = logits.argmax(-1, keepdim=True)
    best = p_teacher.max(-1, keepdim=True).values
    return (p_teacher.gather(-1, pick) >= best * math.exp(-tie_tol / label_temperature) - 1e-7).squeeze(-1)


@torch.no_grad()
def offline_metrics(model, arch: str, data, device, max_frames: int = 20000, batch_size: int = 2048) -> dict:
    from models import policy_outputs
    model.eval()
    idx = data.holdout_idx
    if len(idx) > max_frames:
        g = torch.Generator().manual_seed(0)
        idx = idx[torch.randperm(len(idx), generator=g)[:max_frames]]
    kls, agree, sl, sy, th_ok = [], [], [], [], []
    for a in range(0, len(idx), batch_size):
        b = data.batch(idx[a:a + batch_size], device)
        out = policy_outputs(model, arch, b)
        _, parts = contract_loss(out, b, "mlp")
        kls.append(parts["kl"].item() * len(b["action"]))
        agree.append(top_set_agreement(out["logits"], b["p_teacher"]).float().sum().item())
        if arch != "mlp":
            av = b["act_avail"]
            sl.append(out["safe_logits"][av].float().cpu().numpy())
            sy.append(b["safe"][av].cpu().numpy())
            th_ok.append((out["threat_logits"].argmax(-1) == b["threat"]).float().sum().item())
    n = max(1, len(idx))
    m = {"holdout_frames": int(len(idx)), "kl": sum(kls) / n, "agreement": sum(agree) / n}
    if arch != "mlp":
        m["safe_auc"] = auc(np.concatenate(sl), np.concatenate(sy)) if sl else float("nan")
        m["threat_acc"] = sum(th_ok) / n
    model.train()
    return m


# ------------------------------------------------------------------------------------------
# online evaluation (real env, VAL seeds, planner reference ratio)
# ------------------------------------------------------------------------------------------
def online_eval(model, arch: str, level: str, n_seeds: int, frame_cap: int, cache_dir, workers: int,
                device=None) -> dict:
    """Fly the greedy policy on the first ``n_seeds`` VAL seeds of ``level``; ratio vs the
    planner reference cache (missing entries are computed and merged)."""
    from eval.planner_cache import ensure, ratio
    from models.policy import fly_lockstep
    seeds = seeds_for("val", level)[:n_seeds]
    planner = ensure(cache_dir, "val", level, seeds, frame_cap, workers)
    was_dev = next(model.parameters()).device
    was_training = model.training
    m = model.to("cpu")
    t0 = time.time()
    res = fly_lockstep(m, arch, level, seeds, frame_cap)
    model.to(was_dev)
    model.train(was_training)
    md = {r["seed"]: r["distance_m"] for r in res}
    return {"level": level, "seeds": len(seeds), "frame_cap": frame_cap, "ratio": ratio(md, planner),
            "mean_distance_m": float(np.mean(list(md.values()))),
            "planner_mean_distance_m": float(np.mean([planner[s]["distance_m"] for s in seeds])),
            "survived": int(sum(r["frames"] >= frame_cap for r in res)),
            "crash_causes": {c: sum(1 for r in res if (r["crash_cause"] or "survived") == c)
                             for c in {r["crash_cause"] or "survived" for r in res}},
            "seconds": round(time.time() - t0, 1)}


def final_val(ckpt_path, level: str, n_seeds: int, frame_cap: int, cache_dir, workers: int) -> dict:
    """Evaluate a checkpoint on ``n_seeds`` VAL seeds (all 50 for real runs): the gate number."""
    from models import load_checkpoint
    model, ck = load_checkpoint(ckpt_path)
    r = online_eval(model, ck["arch"], level, n_seeds, frame_cap, cache_dir, workers)
    return {"val_ratio": r["ratio"], "val_level": level, "val_seeds": r["seeds"], "val_frame_cap": frame_cap,
            "val_mean_distance_m": r["mean_distance_m"], "val_planner_mean_distance_m": r["planner_mean_distance_m"],
            "val_survived": r["survived"], "val_crash_causes": r["crash_causes"]}
