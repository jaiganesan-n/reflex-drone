"""Temperature calibration and the hesitation threshold tau.

    uv run python -m train.calibrate --model runs/<run>/best.pt --level L5 --seeds val --out runs/<run>/calibration.json [--max-frames 20000]

1. Fly the model (greedy) on VAL seeds of ``--level`` until ``--max-frames`` frames are
   collected; the teacher labels every visited frame (replan 1, multiprocess).
2. Fit T by minimising the NLL of the teacher's best action(s) under softmax(logits / T)
   (teacher scores often tie; the NLL uses the probability of the whole tied best set, with
   scores within 0.1 of the best counted as tied).
3. ECE with 15 bins before (T = 1) and after.
4. tau = the confidence below which top-1 agreement with the teacher falls under 70%:
   agreement as a function of calibrated confidence is fitted by isotonic regression, and tau
   is the lowest confidence where the fit reaches 70%.

tau marks "hesitation": shown on the HUD, logged, and used to weight DAgger frames. It never
triggers a planner call.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from data.rollout import run_episode
from drone_env.seeds import seeds_for
from teacher.parallel import parallel_map
from teacher.planner import load_teacher_config, soft_targets

AGREE_TARGET = 0.70
N_BINS = 15


def _softmax(z: np.ndarray, T: float) -> np.ndarray:
    s = z / T
    m = np.max(s, axis=-1, keepdims=True)
    e = np.where(np.isfinite(s), np.exp(s - m), 0.0)
    return e / e.sum(-1, keepdims=True)


TIE_TOL = 0.1          # teacher scores within 0.1 (a tenth of a frame survived) count as tied


def best_sets(teacher_scores: np.ndarray, tol: float = TIE_TOL) -> np.ndarray:
    """(N, 5) bool: the teacher's best action(s), counting scores within ``tol`` of the best
    as tied (e.g. ASCEND's 0.05 climb penalty makes it a near-tie with HOLD in open air)."""
    s = np.where(np.isfinite(teacher_scores), teacher_scores, -np.inf)
    return s >= (s.max(-1, keepdims=True) - tol)


def nll(logits: np.ndarray, best: np.ndarray, T: float) -> float:
    p = _softmax(logits, T)
    return float(-np.mean(np.log(np.clip((p * best).sum(-1), 1e-12, 1.0))))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[0]) & (conf <= edges[1])
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def fit_temperature(logits: np.ndarray, best: np.ndarray) -> float:
    """Golden-section search on log T in [log 0.05, log 20]."""
    lo, hi = math.log(0.05), math.log(20.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(logits, best, math.exp(a)), nll(logits, best, math.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll(logits, best, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll(logits, best, math.exp(b))
    return float(math.exp((lo + hi) / 2))


def isotonic(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the best non-decreasing fit to y (already sorted by x)."""
    vals, wts, sizes = [], [], []
    for v in y.astype(np.float64):
        vals.append(v)
        wts.append(1.0)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            n2 = sizes[-2] + sizes[-1]
            vals[-2:], wts[-2:], sizes[-2:] = [v2], [w], [n2]
    return np.repeat(vals, sizes)


def pick_tau(conf: np.ndarray, correct: np.ndarray, target: float = AGREE_TARGET) -> float:
    """tau = the confidence below which top-1 agreement with the teacher falls under ``target``.

    Agreement as a function of confidence is estimated by isotonic regression (monotone in
    confidence); tau is the lowest confidence where it reaches ``target``. 0 if every frame is
    already above target (never hesitant); 1 if agreement never reaches it."""
    if len(conf) == 0:
        return 0.0
    order = np.argsort(conf, kind="mergesort")
    c = conf[order]
    fit = isotonic(correct[order])
    ok = np.flatnonzero(fit >= target)
    if len(ok) == 0:
        return 1.0
    k = int(ok[0])
    return 0.0 if k == 0 else float(c[k])


def calibrate(model: str, level: str, max_frames: int = 20000, workers: int = 6, frame_cap: int = 3000,
              split: str = "val", teacher_config: str | None = None) -> dict:
    if split != "val":
        raise SystemExit("calibration uses VAL seeds only")
    tcfg = load_teacher_config(teacher_config)
    seeds = seeds_for("val", level)
    parts = []
    total = 0
    t0 = time.time()
    i = 0
    while total < max_frames and i < len(seeds):
        chunk = seeds[i:i + max(1, workers)]
        i += len(chunk)
        tasks = [{"level": level, "seed": s, "episode_id": k, "frames_cap": frame_cap, "teacher": tcfg.to_dict(),
                  "burst_prob": 0.0, "student": {"model": str(model), "temperature": 1.0, "beta": 0.0}}
                 for k, s in enumerate(chunk, start=i - len(chunk))]
        res = [None] * len(tasks)
        for j, r in parallel_map(run_episode, tasks, workers):
            res[j] = r
        for r in res:
            parts.append(r)
            total += r["outcome"]["frames"]
        print(f"[calibrate] {i} seeds flown, {total:,} frames  [{time.time() - t0:.0f} s]", flush=True)
    logits = np.concatenate([p["student_logits"] for p in parts])[:max_frames].astype(np.float64)
    scores = np.concatenate([p["teacher_scores"] for p in parts])[:max_frames]
    threat = np.concatenate([p["threat"] for p in parts])[:max_frames].astype(np.int64)
    best = best_sets(scores)
    p1 = _softmax(logits, 1.0)
    correct = best[np.arange(len(p1)), p1.argmax(-1)]
    T = fit_temperature(logits, best)
    pT = _softmax(logits, T)
    conf1, confT = p1.max(-1), pT.max(-1)
    tau = pick_tau(confT, correct)
    out = {
        "model": str(model), "level": level, "seeds": "val", "seeds_flown": i, "frames": int(len(logits)),
        "temperature": T, "tau": tau, "tau_rule": f"isotonic fit of top-1 agreement vs calibrated confidence "
                                                  f"is below {int(AGREE_TARGET * 100)}% under tau",
        "ece_before": ece(conf1, correct), "ece_after": ece(confT, correct),
        "nll_before": nll(logits, best, 1.0), "nll_after": nll(logits, best, T),
        "agreement": float(correct.mean()), "hesitation_share": float((confT < tau).mean()),
        "tie_tolerance": TIE_TOL,
        "by_threat": {str(k): {"frames": int((threat == k).sum()),
                               "agreement": float(correct[threat == k].mean()) if (threat == k).any() else None,
                               "mean_confidence": float(confT[threat == k].mean()) if (threat == k).any() else None,
                               "hesitation_share": float((confT[threat == k] < tau).mean()) if (threat == k).any() else None}
                      for k in range(5)},
        "mean_distance_m": float(np.mean([p["outcome"]["distance_m"] for p in parts])),
        "seconds": round(time.time() - t0, 1),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m train.calibrate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", default="L5")
    ap.add_argument("--seeds", default="val", choices=["val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--frame-cap", type=int, default=3000, help="frames per calibration episode")
    ap.add_argument("--teacher-config", default=None, help="teacher settings file (default configs/train/teacher.yaml)")
    args = ap.parse_args(argv)
    out = calibrate(args.model, args.level, args.max_frames, args.workers, args.frame_cap, args.seeds,
                    args.teacher_config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("temperature", "tau", "ece_before", "ece_after", "agreement",
                                          "hesitation_share", "frames", "seconds")}, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
