"""Planner reference caches: ``<cache-dir>/planner_val.json`` and ``planner_final.json``.

One entry per (level, seed). Each entry records the reference settings (beam 2, replan 5, the
shared teacher.yaml search/scoring settings, the frame cap) and the hash of configs/game.yaml +
configs/levels.yaml. Entries that don't match the current settings are recomputed. New results
are merged into the file; it is never replaced wholesale.
"""
from __future__ import annotations

import json
from pathlib import Path

from drone_env import constants as C
from drone_env.seeds import split_of
from teacher.planner import (REFERENCE_REPLAN, TeacherConfig, fly_many, load_teacher_config, reference_config,
                             reference_settings)

GROUP_SPLITS = {"val": ("val",), "final": ("test", "stress")}


def group_of_split(split: str) -> str:
    return "val" if split == "val" else "final"


def cache_path(cache_dir, group: str) -> Path:
    if group not in GROUP_SPLITS:
        raise ValueError(f"unknown group {group!r}")
    return Path(cache_dir) / f"planner_{group}.json"


def load_cache(path) -> dict:
    path = Path(path)
    if path.exists():
        d = json.loads(path.read_text())
        d.setdefault("entries", {})
        return d
    return {"entries": {}}


def _save(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=1, sort_keys=True))
    tmp.replace(path)


def key(level: str, seed: int) -> str:
    return f"{level}:{int(seed)}"


def valid(entry: dict | None, ref: dict, chash: str) -> bool:
    return bool(entry) and entry.get("ref") == ref and entry.get("config_hash") == chash


def merge(path, new_entries: dict, group: str) -> dict:
    """Re-read the file and merge entries in (never replaces the file's other entries)."""
    path = Path(path)
    cache = load_cache(path)
    cache["group"] = group
    cache["entries"].update(new_entries)
    _save(path, cache)
    return cache


def ensure(cache_dir, group: str, level: str, seeds, frame_cap: int, workers: int = 6,
           teacher_cfg: TeacherConfig | None = None, progress: bool = True) -> dict:
    """Make sure the cache holds valid reference entries for (level, seed) and return them
    as {seed: entry}. Missing or mismatched entries are flown with the fixed reference planner."""
    allowed = GROUP_SPLITS[group]
    for s in seeds:
        if split_of(int(s)) not in allowed:
            raise ValueError(f"seed {s} is not in the {group} group ({allowed})")
    tcfg = teacher_cfg if teacher_cfg is not None else load_teacher_config()
    ref = reference_settings(tcfg, frame_cap)
    chash = C.config_hash()
    path = cache_path(cache_dir, group)
    entries = load_cache(path)["entries"]
    missing = [int(s) for s in seeds if not valid(entries.get(key(level, s)), ref, chash)]
    if missing:
        if progress:
            print(f"[planner-cache] {path}: computing {len(missing)} reference entries for {level} "
                  f"(beam 2, replan {REFERENCE_REPLAN}, cap {frame_cap}) with {workers} workers", flush=True)
        new = {}
        for r in fly_many(level, missing, reference_config(tcfg), REFERENCE_REPLAN, frame_cap, workers,
                          progress=progress):
            e = {k: r[k] for k in ("level", "seed", "distance_m", "frames", "crash_cause", "survived",
                                   "decisions", "decision_ms_mean")}
            e.update({"ref": ref, "config_hash": chash, "spec_version": C.SPEC_VERSION})
            new[key(level, r["seed"])] = e
        entries = merge(path, new, group)["entries"]
    return {int(s): entries[key(level, s)] for s in seeds}


def ratio(model_dist: dict, planner: dict) -> float:
    """Mean model distance / mean planner distance on the same seeds."""
    seeds = sorted(model_dist)
    m = sum(model_dist[s] for s in seeds) / len(seeds)
    p = sum(planner[s]["distance_m"] for s in seeds) / len(seeds)
    return m / p if p > 0 else float("nan")
