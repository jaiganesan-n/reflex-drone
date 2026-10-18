"""Evaluate an agent (or the reference planner) on fixed seed lists (SPEC §10, §16).

    uv run python -m eval.evaluate --agent <spec or planner> --levels <L1..L5 and/or stress> \
        --seeds <val or test> --out reports/<name>.json [--workers 6] [--max-seeds N] [--cache-dir reports]

* Every episode is capped at 6,000 frames. Level ``stress`` always uses STRESS seeds.
* Groups: ``--seeds val`` -> group "val"; ``--seeds test`` (TEST seeds for L1-L5, STRESS for
  stress) -> group "final". This is the only module that reads TEST or STRESS seeds.
* ``--agent planner`` flies the fixed reference (beam 2, replan 5) on the simulator. Its results
  are cached per (level, seed) in ``<cache-dir>/planner_<group>.json`` (merged, never replaced)
  and reused for distance ratios. When ``--out`` is that cache file (as in SPEC §15), the file
  holds both the cache entries and this run's result fields. The planner is not an app agent.
* Agent specs (``onnx-arc:...``, ``onnx-mlp:...``, ``rule``) fly the real env through the agent
  interface, one decision per frame, with ``act()`` timed. Ratios use the planner cache when it
  holds every (level, seed) evaluated.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from drone_env import constants as C
from drone_env.seeds import seeds_for
from eval import planner_cache as pc
from teacher.planner import EVAL_FRAME_CAP, load_teacher_config, reference_settings

_AGENTS: dict = {}


def _agent(spec: str):
    if spec not in _AGENTS:
        from app.agents_base import load_agent
        _AGENTS[spec] = load_agent(spec)
    return _AGENTS[spec]


def fly_agent(task: tuple) -> dict:
    """One episode of an agent spec (worker process)."""
    spec, level, seed, cap = task
    from drone_env.env import DroneDashEnv
    agent = _agent(spec)
    env = DroneDashEnv(level=level, render_mode=None, max_frames=cap)
    env.sensor_spec = getattr(agent, "sensor_spec", C.SENSOR_SPEC)
    obs, info = env.reset(seed=seed)
    agent.reset(env.action_descriptors())
    times = []
    hes = 0
    pc_ = time.perf_counter
    while True:
        t0 = pc_()
        d = agent.act(obs)
        times.append((pc_() - t0) * 1000.0)
        hes += bool(d.hesitating)
        obs, _r, term, trunc, info = env.step(int(d.action))
        if term or trunc:
            break
    t = np.asarray(times)
    return {"level": level, "seed": int(seed), "distance_m": float(info["distance_m"]), "frames": int(info["frame"]),
            "crash_cause": info["crash_cause"], "survived": bool(info["frame"] >= cap and not term),
            "hesitation_share": hes / max(len(times), 1), "masked_action_used": int(info["masked_action_used"]),
            "decision_ms_mean": float(t.mean()), "decision_ms_p50": float(np.percentile(t, 50)),
            "decision_ms_p99": float(np.percentile(t, 99))}


def summarize(rows: list[dict], planner: dict | None) -> dict:
    d = np.array([r["distance_m"] for r in rows])
    out = {"episodes": len(rows), "mean_distance_m": float(d.mean()), "median_distance_m": float(np.median(d)),
           "min_distance_m": float(d.min()), "max_distance_m": float(d.max()),
           "survival_share": float(np.mean([r["survived"] for r in rows])),
           "crash_causes": dict(collections.Counter(r["crash_cause"] or "survived" for r in rows)),
           "hesitation_share": float(np.mean([r.get("hesitation_share", 0.0) for r in rows])),
           "decision_ms_mean": float(np.mean([r["decision_ms_mean"] for r in rows]))}
    if "decision_ms_p99" in rows[0]:
        out["decision_ms_p99_median_over_episodes"] = float(np.median([r["decision_ms_p99"] for r in rows]))
    if planner is not None:
        pm = float(np.mean([planner[r["seed"]]["distance_m"] for r in rows]))
        out["planner_mean_distance_m"] = pm
        out["ratio"] = float(d.mean() / pm) if pm > 0 else None
    return out


def evaluate(agent: str, levels: list[str], split: str, out: str, workers: int = 6, max_seeds: int | None = None,
             cache_dir: str = "reports", frame_cap: int = EVAL_FRAME_CAP) -> dict:
    if split not in ("val", "test"):
        raise SystemExit("--seeds must be val or test")
    if split == "val" and "stress" in levels:
        raise SystemExit("level stress runs on STRESS seeds, which belong to the final group: use --seeds test")
    group = pc.group_of_split(split)
    tcfg = load_teacher_config()
    t0 = time.time()
    per_seed, summary = [], {}
    plan_entries = {}
    for lv in levels:
        seeds = seeds_for(split, lv)[:max_seeds]
        if agent == "planner":
            ents = pc.ensure(cache_dir, group, lv, seeds, frame_cap, workers, tcfg)
            rows = [dict(ents[s], hesitation_share=0.0) for s in seeds]
            plan_entries[lv] = ents
        else:
            from teacher.parallel import parallel_map
            rows = [None] * len(seeds)
            tasks = [(agent, lv, int(s), frame_cap) for s in seeds]
            for i, r in parallel_map(fly_agent, tasks, workers):
                rows[i] = r
            # ratios only from a complete, matching planner cache (never computed here)
            ref = reference_settings(tcfg, frame_cap)
            ents = pc.load_cache(pc.cache_path(cache_dir, group))["entries"]
            got = {s: ents.get(pc.key(lv, s)) for s in seeds}
            if all(pc.valid(e, ref, C.config_hash()) for e in got.values()):
                plan_entries[lv] = got
        per_seed += rows
        summary[lv] = summarize(rows, plan_entries.get(lv))
        s = summary[lv]
        print(f"[eval] {agent} {lv} ({len(rows)} {('STRESS' if lv == 'stress' else split.upper())} seeds): "
              f"mean {s['mean_distance_m']:.1f} m, survived {100 * s['survival_share']:.0f}%"
              + (f", ratio {s['ratio']:.3f}" if s.get("ratio") is not None else ", ratio n/a (planner cache incomplete)")
              + f"  {s['crash_causes']}", flush=True)
    info = None
    if agent != "planner":
        try:
            sc = getattr(_agent(agent), "sidecar", None)
            info = {k: v for k, v in (sc or {}).items() if k != "parity"} if sc else None
        except Exception:
            info = None
    res = {"group": group, "split": split, "levels": levels, "agent": agent, "agent_info": info,
           "frame_cap": frame_cap, "max_seeds": max_seeds, "spec_version": C.SPEC_VERSION,
           "config_hash": C.config_hash(), "reference": reference_settings(tcfg, frame_cap),
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
           "summary": summary, "per_seed": per_seed}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cp = pc.cache_path(cache_dir, group)
    if agent == "planner" and Path(out).resolve() == cp.resolve():
        # SPEC §15: reports/planner_<group>.json is both this result and the cache. Keep every
        # cached entry (merged, never replaced) and add the result fields for this run.
        cache = pc.load_cache(cp)
        cache.update(res)
        tmp = cp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, indent=1, sort_keys=True))
        tmp.replace(cp)
    else:
        Path(out).write_text(json.dumps(res, indent=1))
    print(f"[eval] -> {out}  [{res['seconds']:.0f} s]")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.evaluate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="agent spec (onnx-arc:<path>, onnx-mlp:<path>, rule) or planner")
    ap.add_argument("--levels", nargs="+", required=True, choices=list(C.BASE_LEVELS) + ["stress"])
    ap.add_argument("--seeds", required=True, choices=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-seeds", type=int, default=None)
    ap.add_argument("--cache-dir", default="reports")
    ap.add_argument("--frame-cap", type=int, default=EVAL_FRAME_CAP, help="default 6000 (SPEC §16); smoke uses 600")
    args = ap.parse_args(argv)
    evaluate(args.agent, args.levels, args.seeds, args.out, args.workers, args.max_seeds, args.cache_dir, args.frame_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
