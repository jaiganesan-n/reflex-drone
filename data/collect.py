"""Collect teacher-labelled frames (SPEC §12, §15).

    uv run python -m data.collect --config configs/train/collect.yaml --out data/teacher [--target-frames N]

* The teacher flies episodes on seeds of the configured split (TRAIN; the smoke config uses
  VAL) that this dataset hasn't used yet. With probability ``burst_prob`` per frame a 1-5 frame
  burst of random available actions starts, so the data covers recovery states. Every frame is
  labelled by the teacher (replan 1, settings from configs/train/teacher.yaml).
* Levels follow ``level_mix`` as a share of *frames*: each new episode goes to the level that
  is furthest behind its share (in-flight episodes counted at their level's mean length).
* Workers: multiprocessing, spawn start method. Shards close at the first episode boundary
  at or after 50,000 frames.
* Resumable and top-up-able: finished shards are kept and their seeds recorded in meta.json;
  re-running continues, and a larger --target-frames adds new shards from new seeds. The
  teacher settings and game configs must match the existing shards; burst_prob, frames_cap and
  level_mix may change between top-ups (each shard records its own).
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

import numpy as np
import yaml

from data.dataset import (FIELDS, SHARD_FRAMES, concat, list_shards, read_meta, shard_path, stable_hash,
                          write_meta, write_shard)
from data.rollout import run_episode
from drone_env import constants as C
from drone_env.seeds import SPLITS
from teacher.parallel import make_executor
from teacher.planner import load_teacher_config

DEFAULTS = {"target_frames": 400_000, "workers": 6, "burst_prob": 0.1, "frames_cap": 3000,
            "level_mix": {"L1": 0.10, "L2": 0.15, "L3": 0.20, "L4": 0.25, "L5": 0.30},
            "seeds": "train", "shard_frames": SHARD_FRAMES, "teacher_config": None}


def load_collect_config(path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    cfg = {**DEFAULTS, **raw}
    mix = {k: float(v) for k, v in cfg["level_mix"].items() if float(v) > 0}
    for lv in mix:
        if lv not in C.BASE_LEVELS:
            raise ValueError(f"level_mix: {lv!r} is not one of {C.BASE_LEVELS}")
    tot = sum(mix.values())
    cfg["level_mix"] = {k: v / tot for k, v in mix.items()}
    if cfg["seeds"] not in ("train", "val"):
        raise ValueError("seeds must be train (or val for smoke runs); TEST/STRESS are for final evaluation")
    return cfg


class LevelPicker:
    """Send each new episode to the level furthest behind its share of frames."""

    def __init__(self, mix: dict, cap: int, done: dict | None = None):
        self.mix = mix
        self.frames = {lv: float((done or {}).get(lv, 0)) for lv in mix}
        self.eps = {lv: 0 for lv in mix}
        self.len_sum = {lv: 0.0 for lv in mix}
        self.inflight = {lv: 0 for lv in mix}
        self.cap = cap

    def mean_len(self, lv) -> float:
        return self.len_sum[lv] / self.eps[lv] if self.eps[lv] else float(self.cap)

    def inflight_estimate(self, lv) -> float:
        """Conservative length guess for an unfinished episode (avoids overshooting the target):
        the larger of its level's mean and the mean over all levels (frames_cap until 3 are done)."""
        n = sum(self.eps.values())
        overall = sum(self.len_sum.values()) / n if n >= 3 else float(self.cap)
        return max(self.mean_len(lv), overall)

    def pick(self) -> str:
        proj = {lv: self.frames[lv] + self.inflight[lv] * self.mean_len(lv) for lv in self.mix}
        tot = sum(proj.values()) + 1.0
        lv = max(self.mix, key=lambda l: self.mix[l] - proj[l] / tot)
        self.inflight[lv] += 1
        return lv

    def finished(self, lv: str, frames: int) -> None:
        self.inflight[lv] -= 1
        self.frames[lv] += frames
        self.eps[lv] += 1
        self.len_sum[lv] += frames


def collect(cfg: dict, out: str | Path, target_frames: int | None = None, workers: int | None = None,
            quiet: bool = False, student: dict | None = None, kind: str = "teacher",
            seed_start: int | None = None, exclude_seeds=()) -> dict:
    """Collect labelled frames into ``out`` (see module docstring). ``student`` (DAgger):
    {"model": ckpt, "temperature": T, "beta": b} flies instead of the teacher, mixed at beta."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tcfg = load_teacher_config(cfg.get("teacher_config"))
    target = int(target_frames if target_frames is not None else cfg["target_frames"])
    workers = int(workers if workers is not None else cfg["workers"])
    cap = int(cfg["frames_cap"])
    # what the labels depend on must match across shards; burst_prob, frames_cap and level_mix
    # may change between top-ups (recorded per shard)
    settings = {"teacher": tcfg.to_dict(), "seeds": cfg["seeds"], "config_hash": C.config_hash(),
                "student": student}
    per_shard = {"burst_prob": float(cfg["burst_prob"]), "frames_cap": cap, "level_mix": cfg["level_mix"]}
    meta = read_meta(out)
    if meta is None:
        meta = {"kind": kind, "student": student, "spec_version": C.SPEC_VERSION, "config_hash": C.config_hash(),
                "teacher": tcfg.to_dict(), "teacher_hash": stable_hash(tcfg.to_dict()), "beam": tcfg.beam,
                "collect": {k: cfg[k] for k in ("burst_prob", "frames_cap", "level_mix", "seeds", "shard_frames")},
                "settings_hash": stable_hash(settings), "seeds_used": [], "shards": [], "total_frames": 0,
                "frames_by_level": {}, "episodes": 0}
    elif meta.get("settings_hash") != stable_hash(settings):
        raise SystemExit(f"{out} was labelled with different settings (teacher.yaml, the seed split or "
                         f"configs/game.yaml + levels.yaml). Use a new --out, or restore the settings.")
    # drop orphan shards (written but not recorded: the process died between the two writes)
    known = {s["file"] for s in meta["shards"]}
    for p in list_shards(out):
        if p.name not in known:
            print(f"[collect] removing unrecorded shard {p.name} (interrupted write)")
            p.unlink()
    done_frames = int(meta["total_frames"])
    if done_frames >= target:
        print(f"[collect] {out}: {done_frames:,} frames already >= target {target:,}. Nothing to do.")
        return meta

    split = SPLITS[cfg["seeds"]]
    used = set(meta["seeds_used"]) | {int(x) for x in exclude_seeds}
    start = split.start if seed_start is None else int(seed_start)
    mine = [x for x in meta["seeds_used"] if x >= start]
    next_seed = (max(mine) + 1) if mine else start
    picker = LevelPicker(cfg["level_mix"], cap, meta.get("frames_by_level"))
    shard_k = len(meta["shards"])
    next_ep = int(meta["episodes"])
    buf: list[dict] = []
    buf_frames = 0
    t0 = time.time()
    new_frames = 0
    print(f"[collect] {out}: have {done_frames:,} frames, target {target:,}; workers={workers} "
          f"beam={tcfg.beam} depth={tcfg.depth} burst_prob={cfg['burst_prob']} cap={cap} seeds={cfg['seeds']}",
          flush=True)

    def flush(final: bool = False):
        nonlocal buf, buf_frames, shard_k
        if not buf or (buf_frames < cfg["shard_frames"] and not final):
            return
        arrays = concat(buf)
        seeds = sorted({int(b["outcome"]["seed"]) for b in buf})
        lv_counts = {}
        for b in buf:
            lv = b["outcome"]["level"]
            lv_counts[lv] = lv_counts.get(lv, 0) + b["outcome"]["frames"]
        name = shard_path(out, shard_k).name
        write_shard(out / name, arrays)
        meta["shards"].append({"file": name, "frames": int(buf_frames), "episodes": len(buf), "seeds": seeds,
                               "frames_by_level": lv_counts, **per_shard})
        meta["seeds_used"] = sorted(set(meta["seeds_used"]) | set(seeds))
        meta["total_frames"] = int(meta["total_frames"]) + int(buf_frames)
        meta["episodes"] = int(meta["episodes"]) + len(buf)
        for lv, f in lv_counts.items():
            meta["frames_by_level"][lv] = meta["frames_by_level"].get(lv, 0) + f
        write_meta(out, meta)
        print(f"[collect] wrote {name}: {buf_frames:,} frames, {len(buf)} episodes "
              f"(total {meta['total_frames']:,})", flush=True)
        shard_k += 1
        buf, buf_frames = [], 0

    def make_task():
        nonlocal next_seed, next_ep
        while next_seed in used:
            next_seed += 1
        if next_seed >= split.stop:
            raise SystemExit(f"ran out of {cfg['seeds']} seeds")
        lv = picker.pick()
        task = {"level": lv, "seed": next_seed, "episode_id": next_ep, "frames_cap": cap,
                "teacher": tcfg.to_dict(), "burst_prob": float(cfg["burst_prob"]), "student": student}
        next_seed += 1
        next_ep += 1
        return task

    def handle(res: dict):
        nonlocal buf_frames, new_frames
        o = res.pop("outcome")
        res.pop("executed", None)
        res.pop("student_logits", None)
        res["outcome"] = o
        picker.finished(o["level"], o["frames"])
        buf.append(res)
        buf_frames += o["frames"]
        new_frames += o["frames"]
        el = time.time() - t0
        fps = new_frames / max(el, 1e-6)
        remaining = max(0, target - done_frames - new_frames)
        if not quiet:
            print(f"  ep {o['episode_id']:5d} {o['level']} seed {o['seed']:6d}: {o['frames']:5d} frames "
                  f"{o['crash_cause'] or 'capped'}  |  {done_frames + new_frames:,}/{target:,} frames  "
                  f"{fps:,.0f} frames/s  ETA {remaining / max(fps, 1e-6) / 60:.1f} min", flush=True)
        flush()

    if workers <= 1:
        while done_frames + new_frames < target:
            handle(run_episode(make_task()))
    else:
        with make_executor(workers) as ex:
            inflight = {}
            window = 2 * workers
            while True:
                projected = done_frames + new_frames + sum(picker.inflight_estimate(t["level"])
                                                           for t in inflight.values())
                while len(inflight) < window and projected < target:
                    t = make_task()
                    inflight[ex.submit(run_episode, t)] = t
                    projected += picker.inflight_estimate(t["level"])
                if not inflight:
                    break
                fin, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for f in fin:
                    inflight.pop(f)
                    handle(f.result())
    flush(final=True)
    el = time.time() - t0
    meta.setdefault("timing", []).append({"frames": new_frames, "seconds": round(el, 1), "workers": workers,
                                          "frames_per_s": round(new_frames / max(el, 1e-6), 1)})
    write_meta(out, meta)
    print(f"[collect] done: {meta['total_frames']:,} frames in {len(meta['shards'])} shards; "
          f"{new_frames:,} new frames in {el:.0f} s = {new_frames / max(el, 1e-6):,.0f} frames/s", flush=True)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data.collect", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train/collect.yaml")
    ap.add_argument("--out", default="data/teacher")
    ap.add_argument("--target-frames", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None, help="override the config's workers")
    ap.add_argument("--quiet", action="store_true", help="no per-episode lines")
    args = ap.parse_args(argv)
    collect(load_collect_config(args.config), args.out, args.target_frames, args.workers, args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
