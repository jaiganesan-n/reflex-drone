"""Speed benchmark on one core (SPEC §7 targets).

    uv run python -m drone_env.bench

1. env.step (with observation): the rule agent flies L5 VAL seeds through gymnasium.
2. Simulator.step (no observation): the recorded actions are replayed on bare Simulators,
   so the sim follows exactly the same trajectories (observing never changes the world).
3. clone(calm=True): timed on states sampled along those trajectories.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

TARGET_SIM = 25_000
TARGET_ENV = 5_000
TARGET_CLONE_US = 50.0


def run(level: str = "L5", n_seeds: int = 12, cap: int = 3000) -> dict:
    import gymnasium as gym
    import drone_env  # noqa: F401
    from app.agents_base import RuleAgent
    from drone_env.seeds import seeds_for
    from drone_env.sim import Simulator

    seeds = seeds_for("val", level)[:n_seeds]
    agent = RuleAgent()
    episodes = []
    env_time = env_steps = act_time = 0.0
    for seed in seeds:
        env = gym.make("DroneDash-v0", level=level, max_frames=cap).unwrapped
        agent.reset(env.action_descriptors())
        obs, _ = env.reset(seed=seed)
        actions = []
        while True:
            t0 = time.perf_counter()
            a = agent.act(obs).action
            t1 = time.perf_counter()
            obs, r, term, trunc, info = env.step(a)
            t2 = time.perf_counter()
            act_time += t1 - t0
            env_time += t2 - t1
            env_steps += 1
            actions.append(a)
            if term or trunc:
                break
        episodes.append((seed, actions))

    # replay on bare simulators (no observe)
    sim_time = sim_steps = 0.0
    states = []
    for seed, actions in episodes:
        sim = Simulator(level, seed)
        step = sim.step
        t0 = time.perf_counter()
        for a in actions:
            step(a)
        sim_time += time.perf_counter() - t0
        sim_steps += len(actions)
        # sample states for clone timing
        sim = Simulator(level, seed)
        for i, a in enumerate(actions):
            if i % 50 == 25:
                states.append(sim.clone(calm=False))
            sim.step(a)

    clone_us = []
    for s in states:
        for _ in range(20):
            t0 = time.perf_counter()
            s.clone(calm=True)
            clone_us.append((time.perf_counter() - t0) * 1e6)
    obs_us = []
    for s in states[:200]:
        t0 = time.perf_counter()
        s.observe()
        obs_us.append((time.perf_counter() - t0) * 1e6)
    clr_us = []
    for s in states[:200]:
        t0 = time.perf_counter()
        s.clearance()
        clr_us.append((time.perf_counter() - t0) * 1e6)
    return {
        "level": level, "seeds": len(seeds), "frames": int(sim_steps),
        "sim_steps_per_s": sim_steps / sim_time,
        "env_steps_per_s": env_steps / env_time,
        "clone_us_mean": float(np.mean(clone_us)), "clone_us_p99": float(np.percentile(clone_us, 99)),
        "observe_us_mean": float(np.mean(obs_us)), "clearance_us_mean": float(np.mean(clr_us)),
        "rule_act_us_mean": act_time / env_steps * 1e6,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="L5")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--cap", type=int, default=3000)
    args = ap.parse_args()
    r = run(args.level, args.seeds, args.cap)
    ok = lambda b: "PASS" if b else "FAIL"   # noqa: E731
    print(f"DroneDash bench · level {r['level']} · {r['seeds']} VAL seeds · {r['frames']:,} frames · one core")
    print(f"  Simulator.step (no obs)  {r['sim_steps_per_s']:>10,.0f} steps/s   target >= {TARGET_SIM:,}   "
          f"{ok(r['sim_steps_per_s'] >= TARGET_SIM)}")
    print(f"  env.step (with obs)      {r['env_steps_per_s']:>10,.0f} steps/s   target >= {TARGET_ENV:,}    "
          f"{ok(r['env_steps_per_s'] >= TARGET_ENV)}")
    print(f"  clone(calm=True)         {r['clone_us_mean']:>10.1f} us mean ({r['clone_us_p99']:.1f} p99)   "
          f"target <= {TARGET_CLONE_US:.0f} us   {ok(r['clone_us_mean'] <= TARGET_CLONE_US)}")
    print(f"  observe()                {r['observe_us_mean']:>10.1f} us mean")
    print(f"  clearance()              {r['clearance_us_mean']:>10.1f} us mean")
    print(f"  rule agent act()         {r['rule_act_us_mean']:>10.1f} us mean")


if __name__ == "__main__":
    main()
