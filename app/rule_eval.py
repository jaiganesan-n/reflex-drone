"""Fly an agent on VAL seeds and print distances (gate G1 helper).

    uv run python -m app.rule_eval                      # rule on L1 VAL, 3,000-frame cap
    uv run python -m app.rule_eval --levels L1 L3 L5 --max-seeds 10
"""
from __future__ import annotations

import argparse
import collections
import time

import gymnasium as gym
import numpy as np

import drone_env  # noqa: F401  (registers DroneDash-v0)
from app.agents_base import load_agent
from drone_env.seeds import seeds_for


def fly(agent_spec: str, level: str, seed: int, cap: int) -> dict:
    env = gym.make("DroneDash-v0", level=level, max_frames=cap)
    agent = load_agent(agent_spec)
    agent.reset(env.unwrapped.action_descriptors())
    obs, info = env.reset(seed=seed)
    while True:
        obs, r, term, trunc, info = env.step(agent.act(obs).action)
        if term or trunc:
            break
    env.close()
    return {"seed": seed, "distance_m": info["distance_m"], "frames": info["frame"],
            "crash_cause": info["crash_cause"]}


def evaluate(agent_spec: str, level: str, cap: int = 3000, max_seeds: int | None = None) -> list[dict]:
    seeds = seeds_for("val", level)[:max_seeds]
    return [fly(agent_spec, level, s, cap) for s in seeds]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="rule")
    ap.add_argument("--levels", nargs="+", default=["L1"])
    ap.add_argument("--cap", type=int, default=3000, help="frame cap per episode")
    ap.add_argument("--max-seeds", type=int, default=None)
    args = ap.parse_args()
    for level in args.levels:
        t0 = time.time()
        res = evaluate(args.agent, level, args.cap, args.max_seeds)
        d = np.array([r["distance_m"] for r in res])
        causes = collections.Counter(r["crash_cause"] or "survived" for r in res)
        print(f"{args.agent} {level}: mean {d.mean():.1f} m  median {np.median(d):.1f} m  "
              f"min {d.min():.1f} m  over {len(res)} VAL seeds (cap {args.cap} frames)  "
              f"{dict(causes)}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
