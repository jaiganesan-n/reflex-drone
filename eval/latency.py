"""Decision latency on one thread (gate G6: full ``act()`` p99 < 1 ms on the Mac).

    uv run python -m eval.latency --agent onnx-arc:artifacts/arc.onnx --calls 10000 [--eager] --out reports/latency_arc.json

Times 10,000 single-frame calls on real observations (flown by the rule agent on L5 VAL seeds):
* ``act_ms``: the full ``agent.act(obs)`` (building inputs, ONNX Runtime, softmax, Decision);
* ``session_run_ms``: ``session.run`` alone on the same, already-built inputs;
* ``--eager``: also the PyTorch checkpoint named in the sidecar's ``trained_from`` (forward
  only, and inputs + forward), with torch on one thread.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np


def obs_pool(n: int = 3000, level: str = "L5") -> tuple[list, np.ndarray]:
    from app.agents_base import load_agent
    from drone_env.env import DroneDashEnv
    rule = load_agent("rule")
    pool = []
    seed = 100_000
    while len(pool) < n:
        env = DroneDashEnv(level, max_frames=n)
        obs, _ = env.reset(seed=seed)
        rule.reset(env.action_descriptors())
        while len(pool) < n:
            pool.append({k: v.copy() for k, v in obs.items()})
            obs, _, term, trunc, _ = env.step(rule.act(obs).action)
            if term or trunc:
                break
        seed += 1
    return pool, env.action_descriptors()


def stats(ms) -> dict:
    a = np.asarray(ms)
    return {"p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)), "mean": float(a.mean()),
            "max": float(a.max()), "calls": int(len(a))}


def measure(spec: str, calls: int = 10000, eager: bool = False, warmup: int = 300) -> dict:
    from app.agents_base import load_agent
    pool, desc = obs_pool()
    agent = load_agent(spec)
    agent.reset(desc)
    pc = time.perf_counter_ns
    for i in range(warmup):
        agent.act(pool[i % len(pool)])
    act = []
    for i in range(calls):
        o = pool[i % len(pool)]
        t0 = pc()
        agent.act(o)
        act.append((pc() - t0) / 1e6)
    out = {"agent": spec, "calls": calls, "threads": 1, "act_ms": stats(act)}
    if hasattr(agent, "sess"):
        agent.reset(desc)
        run = []
        for i in range(warmup):
            agent.fill(pool[i % len(pool)])
            agent.run()
        for i in range(calls):
            agent.fill(pool[i % len(pool)])
            t0 = pc()
            agent.run()
            run.append((pc() - t0) / 1e6)
        out["session_run_ms"] = stats(run)
    if eager:
        import torch
        from models import load_checkpoint, policy_outputs
        from models.policy import obs_batch
        from drone_env.wrappers import FlatHistory
        torch.set_num_threads(1)
        src = (getattr(agent, "sidecar", None) or {}).get("trained_from")
        if not src or not Path(src).exists():
            out["eager_error"] = f"checkpoint from the sidecar's trained_from not found: {src}"
        else:
            model, ck = load_checkpoint(src)
            arch = ck["arch"]
            hist = FlatHistory() if arch == "mlp" else None
            fwd, full = [], []
            with torch.inference_mode():
                for i in range(warmup + calls):
                    o = pool[i % len(pool)]
                    t0 = pc()
                    flat = hist.push(o)[None] if hist is not None else None
                    b = obs_batch([o], desc, arch, flat)
                    t1 = pc()
                    policy_outputs(model, arch, b)
                    t2 = pc()
                    if i >= warmup:
                        fwd.append((t2 - t1) / 1e6)
                        full.append((t2 - t0) / 1e6)
            out["eager_checkpoint"] = src
            out["eager_forward_ms"] = stats(fwd)
            out["eager_inputs_and_forward_ms"] = stats(full)
            out["torch_version"] = torch.__version__
    import onnxruntime as ort
    out.update({"g6_act_p99_under_1ms": out["act_ms"]["p99"] < 1.0, "machine": platform.machine(),
                "platform": platform.platform(), "processor": platform.processor(),
                "python": platform.python_version(), "onnxruntime_version": ort.__version__,
                "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.latency", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--calls", type=int, default=10000)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    r = measure(args.agent, args.calls, args.eager)
    line = f"[latency] {args.agent}: act() p50 {r['act_ms']['p50']:.3f} ms  p99 {r['act_ms']['p99']:.3f} ms"
    if "session_run_ms" in r:
        line += f"  |  session.run p50 {r['session_run_ms']['p50']:.3f}  p99 {r['session_run_ms']['p99']:.3f} ms"
    if "eager_forward_ms" in r:
        line += f"  |  eager forward p50 {r['eager_forward_ms']['p50']:.3f}  p99 {r['eager_forward_ms']['p99']:.3f} ms"
    print(line)
    print(f"[latency] G6 (act p99 < 1 ms): {'PASS' if r['g6_act_p99_under_1ms'] else 'FAIL'} on {r['platform']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(r, indent=2))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
