"""Results report from evaluation JSONs of ONE group (all val, or all final = test + stress).

    uv run python -m eval.report reports/planner_final.json reports/arc.json reports/mlp.json ... --out reports/results.md

Builds: distance ratio per level for every model, stress drop = 1 - ratio(stress) / ratio(L5),
latency (from ``latency_*.json`` next to --out), the ablation table (planner, MLP, Arc
supervised only, Arc + DAgger, Arc + PPO if present), hesitation share and crash causes.
Refuses to mix the val and final groups.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

from drone_env import constants as C

LEVEL_ORDER = ["L1", "L2", "L3", "L4", "L5", "stress"]
STAGE_LABEL = {"supervised": "Arc supervised only", "dagger": "Arc + DAgger", "ppo": "Arc + PPO"}


class MixedGroups(ValueError):
    pass


def load_results(paths) -> list[dict]:
    res = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        if "group" not in d or "per_seed" not in d:
            raise ValueError(f"{p} is not an eval.evaluate result")
        d["_path"] = str(p)
        res.append(d)
    groups = {d["group"] for d in res}
    if len(groups) > 1:
        raise MixedGroups("refusing to mix groups: " + ", ".join(f"{d['_path']} = {d['group']}" for d in res)
                          + ". Report val and final results separately.")
    return res


def label(d: dict) -> str:
    if d["agent"] == "planner":
        return "planner (reference)"
    spec = d["agent"]
    stem = Path(spec.partition(":")[2]).stem if ":" in spec else spec
    info = d.get("agent_info") or {}
    if info.get("arch") == "mlp" or spec.startswith("onnx-mlp"):
        return f"MLP ({stem})"
    st = info.get("stage")
    return f"{STAGE_LABEL.get(st, 'Arc')} ({stem})" if spec.startswith("onnx-arc") else stem


def per_level(d: dict) -> dict:
    out = collections.defaultdict(dict)
    for r in d["per_seed"]:
        out[r["level"]][r["seed"]] = r
    return out


def ratios(model: dict, planner: dict | None) -> dict:
    """Ratio per level on the seeds both evaluated (planner JSON if given, else the model's own)."""
    out = {}
    mp = per_level(model)
    pp = per_level(planner) if planner else {}
    for lv, rows in mp.items():
        if planner and lv in pp:
            seeds = sorted(set(rows) & set(pp[lv]))
            if seeds:
                m = sum(rows[s]["distance_m"] for s in seeds) / len(seeds)
                p = sum(pp[lv][s]["distance_m"] for s in seeds) / len(seeds)
                out[lv] = (m / p if p > 0 else None, len(seeds))
                continue
        r = (model.get("summary") or {}).get(lv, {}).get("ratio")
        out[lv] = (r, len(rows))
    return out


def fmt(x, nd=3, pct=False):
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}"


def build(paths, out: str, latency_dir: str | None = None) -> str:
    res = load_results(paths)
    group = res[0]["group"]
    planners = [d for d in res if d["agent"] == "planner"]
    planner = planners[0] if planners else None
    models = [d for d in res if d["agent"] != "planner"]
    levels = [lv for lv in LEVEL_ORDER if any(lv in d["levels"] for d in res)]
    L = [f"# Reflex Drone results ({'FINAL group: TEST + STRESS seeds' if group == 'final' else 'VAL group: development numbers, not final'})",
         "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} from: " + ", ".join(f"`{d['_path']}`" for d in res), ""]
    if not planner:
        L += ["> No planner JSON given: ratios come from the planner cache recorded in each file (n/a if missing).", ""]
    rat = {label(d): ratios(d, planner) for d in models}
    # headline
    arc_models = [d for d in models if d["agent"].startswith("onnx-arc")]
    head = next((d for d in arc_models if Path(d["agent"].partition(":")[2]).stem == "arc"), None)
    if head is None and arc_models:
        head = max(arc_models, key=lambda d: (rat[label(d)].get("L5", (None,))[0] or -1))
    lat = {}
    ldir = Path(latency_dir) if latency_dir else Path(out).parent
    for p in sorted(ldir.glob("latency_*.json")):
        try:
            lat[json.loads(p.read_text())["agent"]] = (p, json.loads(p.read_text()))
        except Exception:
            pass
    if head is not None:
        r5 = rat[label(head)].get("L5", (None, 0))[0]
        lp = lat.get(head["agent"], (None, {}))[1].get("act_ms", {}).get("p99")
        L += ["## Headline", "",
              f"**Solo System 1 ({label(head)}): distance ratio {fmt(r5)} on L5 "
              f"{'TEST' if group == 'final' else 'VAL'}, `act()` p99 {fmt(lp)} ms.**", ""]
    # distance ratio table
    L += ["## Distance ratio per level (model mean distance / planner mean distance, same seeds)", "",
          "| model | " + " | ".join(levels) + " |", "|---|" + "---|" * len(levels)]
    if planner:
        L.append("| planner (reference) | " + " | ".join("1.000" if lv in planner["levels"] else "" for lv in levels) + " |")
    for d in models:
        r = rat[label(d)]
        L.append(f"| {label(d)} | " + " | ".join(fmt(r[lv][0]) if lv in r else "" for lv in levels) + " |")
    L += ["", "Mean distance (m):", "", "| model | " + " | ".join(levels) + " |", "|---|" + "---|" * len(levels)]
    for d in ([planner] if planner else []) + models:
        s = d["summary"]
        L.append(f"| {label(d)} | " + " | ".join(f"{s[lv]['mean_distance_m']:.1f}" if lv in s else "" for lv in levels) + " |")
    # stress drop
    L += ["", "## Stress drop (1 - ratio(stress) / ratio(L5); G6 needs <= 0.20)", "", "| model | ratio L5 | ratio stress | stress drop |",
          "|---|---|---|---|"]
    for d in models:
        r = rat[label(d)]
        r5, rs = r.get("L5", (None,))[0], r.get("stress", (None,))[0]
        drop = (1 - rs / r5) if (r5 and rs is not None) else None
        L.append(f"| {label(d)} | {fmt(r5)} | {fmt(rs)} | {fmt(drop)} |")
    # latency
    L += ["", "## Decision latency (one thread; G6 needs `act()` p99 < 1 ms)", ""]
    if lat:
        L += ["| agent | act() p50 | act() p99 | session.run p50 | session.run p99 | eager p50 | eager p99 | machine |",
              "|---|---|---|---|---|---|---|---|"]
        for spec, (p, x) in lat.items():
            a, s, e = x.get("act_ms", {}), x.get("session_run_ms", {}), x.get("eager_forward_ms", {})
            L.append(f"| `{spec}` | {fmt(a.get('p50'))} | {fmt(a.get('p99'))} | {fmt(s.get('p50'))} | {fmt(s.get('p99'))} | "
                     f"{fmt(e.get('p50'))} | {fmt(e.get('p99'))} | {x.get('machine', '')} |")
        L.append("")
        L.append("All times in ms. " + ", ".join(f"`{p.name}`" for p, _ in lat.values()))
    else:
        L.append(f"No `latency_*.json` files in `{ldir}`.")
    # ablation
    L += ["", "## Ablation", "", "| system | ratio L5 | ratio stress | stress drop | hesitation share (L5) |", "|---|---|---|---|---|"]
    if planner:
        L.append("| planner (System 2 reference, beam 2, replan 5) | 1.000 | 1.000 | 0.000 | n/a |")
    order = {"mlp": 0, "supervised": 1, "dagger": 2, "ppo": 3}

    def rank(d):
        info = d.get("agent_info") or {}
        return order.get("mlp" if d["agent"].startswith("onnx-mlp") else info.get("stage"), 9)
    for d in sorted(models, key=rank):
        r = rat[label(d)]
        r5, rs = r.get("L5", (None,))[0], r.get("stress", (None,))[0]
        drop = (1 - rs / r5) if (r5 and rs is not None) else None
        h = d["summary"].get("L5", {}).get("hesitation_share")
        L.append(f"| {label(d)} | {fmt(r5)} | {fmt(rs)} | {fmt(drop)} | {fmt(h, pct=True)} |")
    # hesitation
    L += ["", "## Hesitation share (frames with max probability < tau)", "",
          "| model | " + " | ".join(levels) + " |", "|---|" + "---|" * len(levels)]
    for d in models:
        s = d["summary"]
        L.append(f"| {label(d)} | " + " | ".join(fmt(s[lv].get("hesitation_share"), pct=True) if lv in s else "" for lv in levels) + " |")
    # crash causes
    L += ["", "## Outcomes (all evaluated levels)", ""]
    seen = {r["crash_cause"] or "survived" for d in ([planner] if planner else []) + models for r in d["per_seed"]}
    causes = ["survived"] + [c for c in C.CRASH_CAUSES if c in seen or c in ("ground", "ceiling", "building", "bird",
                                                                         "traffic", "storm")]
    L += ["| model | " + " | ".join(causes) + " |", "|---|" + "---|" * len(causes)]
    for d in ([planner] if planner else []) + models:
        c = collections.Counter(r["crash_cause"] or "survived" for r in d["per_seed"])
        L.append(f"| {label(d)} | " + " | ".join(str(c.get(k, 0)) for k in causes) + " |")
    L += ["", f"Episodes are capped at {res[0].get('frame_cap', 6000)} frames. Seeds: "
          + ("TEST 200,000-200,099 for L1-L5, STRESS 300,000-300,099 for stress." if group == "final"
             else "VAL 100,000-100,049."), ""]
    text = "\n".join(L)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="eval.evaluate JSON files from one group")
    ap.add_argument("--out", default="reports/results.md")
    ap.add_argument("--latency-dir", default=None, help="where latency_*.json live (default: --out's folder)")
    args = ap.parse_args(argv)
    try:
        text = build(args.results, args.out, args.latency_dir)
    except MixedGroups as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(text)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
