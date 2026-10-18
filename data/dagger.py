"""DAgger: the student flies, the teacher corrects (SPEC §15).

    uv run python -m data.dagger --config configs/train/dagger.yaml --init runs/arc_A/best.pt \
        --calibration runs/arc_A/calibration.json --run arc_B

Each round k = 1..rounds:

1. The current student flies TRAIN seeds (new ones each round), mixed with the teacher: each
   frame the teacher's action is executed with probability beta[k], else the student's. Every
   visited state is recorded with ``student_action`` and ``student_conf`` (max probability
   after the calibration temperature) and labelled by the teacher (replan 1, multiprocess).
   Data: ``runs/<run>/round_k/data/``.
2. Retrain from a warm start (the previous round's best.pt) on the base data plus every round
   so far, for ``epochs_per_round`` epochs; frames with student_conf < tau count x2. The online
   ratio (10 VAL seeds of L5) picks the round's best.pt: ``runs/<run>/round_k/``.

Resumable: a round with ``round_k/summary.json`` is skipped; a half-collected round resumes.
Raising ``rounds`` adds rounds. At the end the best round's best.pt is copied to
``runs/<run>/best.pt`` and flown on all 50 VAL seeds of L5 -> ``runs/<run>/summary.json``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

from data.collect import DEFAULTS as COLLECT_DEFAULTS
from data.collect import collect
from data.dataset import read_meta
from train.common import SUPERVISED_DEFAULTS, Run, as_list, final_val, load_config

DAGGER_DEFAULTS = {
    "rounds": 3, "frames_per_round": 60000, "beta": [0.5, 0.25, 0.0], "base_data": "data/teacher",
    "epochs_per_round": 3, "workers": 6, "train_config": "configs/train/arc.yaml", "gate_level": "L5",
    "level_mix": {"L1": 0.05, "L2": 0.10, "L3": 0.20, "L4": 0.30, "L5": 0.35}, "frames_cap": 6000,
    "seed_base": 50_000, "seeds_per_round": 5_000, "seeds": "train", "online_seeds": 10,
    "eval_frame_cap": 6000, "final_val_seeds": 50, "planner_cache_dir": "reports",
    "teacher_config": None,     # teacher settings file for the labels (default configs/train/teacher.yaml)
}


def beta_for(cfg: dict, k: int) -> float:
    b = cfg["beta"]
    if isinstance(b, (int, float)):
        return float(b)
    return float(b[min(k - 1, len(b) - 1)])


def run_dagger(cfg: dict, init: str, calibration: str | None, run_name: str) -> dict:
    run = Run(run_name, {**cfg, "init": init, "calibration": calibration}, fresh_metrics=False)
    T, tau = 1.0, None
    if calibration:
        cal = json.loads(Path(calibration).read_text())
        T, tau = float(cal["temperature"]), float(cal["tau"])
    tcfg = load_config(cfg["train_config"], SUPERVISED_DEFAULTS)
    for k in ("gate_level", "online_seeds", "eval_frame_cap", "planner_cache_dir", "workers"):
        tcfg[k] = cfg[k]
    tcfg["epochs"] = int(cfg["epochs_per_round"])
    tcfg["stage"] = "dagger"
    base = as_list(cfg["base_data"])
    exclude = set()
    for d in base:
        m = read_meta(d)
        if m:
            exclude |= set(m.get("seeds_used", []))
    prev = init
    rounds = []
    t_all = time.time()
    for k in range(1, int(cfg["rounds"]) + 1):
        rdir = run.dir / f"round_{k}"
        rdata = rdir / "data"
        summ = rdir / "summary.json"
        if summ.exists():
            s = json.loads(summ.read_text())
            print(f"[dagger] round {k}: done (online ratio {s.get('best_online_ratio')}); skipping", flush=True)
            rounds.append(s)
            prev = str(rdir / "best.pt")
            continue
        beta = beta_for(cfg, k)
        t0 = time.time()
        print(f"[dagger] round {k}: student {prev} (T={T:.3f}) flies with beta={beta}; "
              f"{cfg['frames_per_round']:,} frames -> {rdata}", flush=True)
        ccfg = {**COLLECT_DEFAULTS, "burst_prob": 0.0, "frames_cap": int(cfg["frames_cap"]),
                "teacher_config": cfg["teacher_config"],
                "level_mix": cfg["level_mix"], "seeds": cfg["seeds"], "workers": int(cfg["workers"])}
        tot = sum(cfg["level_mix"].values())
        ccfg["level_mix"] = {a: float(b) / tot for a, b in cfg["level_mix"].items() if float(b) > 0}
        student = {"model": str(prev), "temperature": T, "beta": beta}
        collect(ccfg, rdata, target_frames=int(cfg["frames_per_round"]), workers=int(cfg["workers"]), quiet=True,
                student=student, kind="dagger", seed_start=int(cfg["seed_base"]) + (k - 1) * int(cfg["seeds_per_round"]),
                exclude_seeds=exclude)
        t_collect = time.time() - t0
        from train.supervised import train_supervised
        dirs = base + [str(run.dir / f"round_{j}" / "data") for j in range(1, k + 1)]
        s = train_supervised(tcfg, dirs, f"{run_name}/round_{k}", init=prev, tau=tau, do_final_val=False)
        s.update({"round": k, "beta": beta, "collect_seconds": round(t_collect, 1),
                  "round_seconds": round(time.time() - t0, 1)})
        (rdir / "summary.json").write_text(json.dumps(s, indent=2))
        run.log({"round": k, "beta": beta, "best_online_ratio": s["best_online_ratio"], "best_epoch": s["best_epoch"],
                 "collect_seconds": s["collect_seconds"], "round_seconds": s["round_seconds"],
                 "train_frames": s["train_frames"]})
        rounds.append(s)
        prev = str(rdir / "best.pt")
    # final: best round by online ratio
    best_k = max(range(len(rounds)), key=lambda i: (rounds[i].get("best_online_ratio") or -1, i))
    best_round = best_k + 1
    shutil.copyfile(run.dir / f"round_{best_round}" / "best.pt", run.path("best.pt"))
    shutil.copyfile(run.dir / f"round_{len(rounds)}" / "last.pt", run.path("last.pt"))
    summary = {"run": run_name, "arch": "arc", "stage": "dagger", "init": init, "calibration": calibration,
               "rounds": len(rounds), "best_round": best_round,
               "round_online_ratios": [r.get("best_online_ratio") for r in rounds],
               "temperature": T, "tau": tau, "seconds": round(time.time() - t_all, 1)}
    summary.update(final_val(run.path("best.pt"), cfg["gate_level"], int(cfg["final_val_seeds"]),
                             int(cfg["eval_frame_cap"]), cfg["planner_cache_dir"], int(cfg["workers"])))
    run.write_summary(summary)
    print(f"[dagger] best round {best_round}; best.pt on {summary['val_seeds']} VAL seeds of {summary['val_level']}: "
          f"val_ratio {summary['val_ratio']:.3f}", flush=True)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data.dagger", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train/dagger.yaml")
    ap.add_argument("--init", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, DAGGER_DEFAULTS)
    run_dagger(cfg, args.init, args.calibration, args.run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
