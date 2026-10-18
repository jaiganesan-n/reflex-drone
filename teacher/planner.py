"""System 2 teacher: beam search on calm clones (SPEC §11).

    uv run python -m teacher.planner --check --level L5 --seeds val [--beam K] [--workers 6] [--max-seeds N]

Search (settings in configs/train/teacher.yaml)
------
* Macro action = one action held for ``macro_frames`` (5) frames, mask permitting (a masked
  action becomes HOLD for that frame). ``depth`` macros per plan (SPEC v1.4: 9 = 1.5 s).
* One beam of width K per available first action: the first macro is fixed, then each depth
  expands every beam node with every action available at that node and keeps the best K.
  Crashed branches are pruned (their score still counts as a finished plan).
* score = frames survived + 0.5 * mean clearance (sampled at the end of each macro, capped at
  2 m, 0 after a crash) - 0.05 * #ASCEND/JUMP macros - jump_extra_penalty * #JUMP macros
  - duck_extra_penalty * #DUCK macros (SPEC v1.7: DUCK is a dive, the mirror of JUMP).
* ``hold_gust``: the root clone keeps the gust the drone feels now (``clone(hold_gust=True)``).
* Optional, all off by default (teacher.yaml, and so the evaluation reference, leaves them off):
  ``descend_penalty`` per DESCEND/DUCK macro (the mirror of climb_penalty); ``slow_burst_extra``
  added to every JUMP/DUCK macro at 8 m/s and fading linearly to 0 at 16 m/s (root speed), so a
  burst at low speed is worth it only when slower moves fail; ``sensor_limited``: the root clone
  keeps only the hazards the drone's sensors show now (nose + 20 m, 16 m before v2.3, 8 m in fog) and spawns
  nothing new, so the teacher never reacts to what the pilot cannot see; ``keep_sustained_moves``:
  a beam node whose last macro was ASCEND or DESCEND always keeps its child that repeats it, on
  top of the best K (pruning by partial score alone drops every steady climb, because each extra
  climbing macro costs a little before the obstacle is reached, so only a JUMP gets over a tall
  building); ``bird_envelope``: every bird is planned as the whole band its up-and-down path
  covers (centre +- (amplitude + half its height), level flight), not its exact sine path, which
  the pilot cannot know from one frame, so the teacher keeps clear of birds instead of threading
  between them; ``path_weight`` (v2.0): + path_weight * mean over the macros of
  min(path_time, H) / H, where path_time is how long the drone could fly level at its height
  (+- path_margin_m) before reaching a hazard (``Simulator.path_time``, H = path_horizon_s): a plan
  that clears the way early beats one that holds until the last moment; ``parts_margin_m`` (v2.0):
  every obstacle built from boxes and wires (trees, street lights, power lines, kites, cranes)
  is planned grown by that margin on every side, so the teacher keeps that distance and does not
  start down over the far end of a long wire or jib, an end the pilot's rays (60 degrees down at
  most) lose sight of before the drone's tail has passed it; ``tracked_movers`` (v2.3): every
  bird, drone, balloon and plane the sensors show is planned as the band the object tracker
  predicts for it while the drone passes it (``drone_env.tracker``, the band in the pilot's v2.3
  hazard row), level flight, instead of its true path: the teacher knows only what the pilot's
  tracker has seen (replaces ``bird_envelope``, which used the game's own path settings);
  ``tracked_fallback`` (v2.3, with tracked_movers): when no first action is safe against the
  tracker's bands, plan again against its best-guess bands (no margins) and use that plan. The
  margins cover most misses; when they box the drone in (a band edge drifting onto it near
  contact), the teacher trusts its best estimate instead of a desperate burst into the ceiling.

Per first action ``a``: ``scores[a]`` = best plan score (-inf if masked), ``safe[a]`` = some
searched plan starting with ``a`` survives ``safe_frames`` (30 = 1 s). Teacher action = argmax
(lowest index on ties). Threat comes from holding on the same calm clone: 0 (> 2 s),
1 (1-2 s), 2 (0.5-1 s), 3 (< 0.5 s), 4 (no safe action).

``plan(sim, cfg)`` is a pure function of a Simulator: it only touches clones, so worker processes
can call it. The hot path is ``Simulator.clone`` and ``Simulator.step``.
"""
from __future__ import annotations

import argparse
import collections
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from drone_env import constants as C
from drone_env.generator import BOX, Mover, Static
from drone_env.sim import Simulator

ROOT = Path(__file__).resolve().parents[1]
TEACHER_YAML = ROOT / "configs" / "train" / "teacher.yaml"

HOLD, ASCEND, DESCEND, JUMP, DUCK = 0, 1, 2, 3, 4
NEG_INF = float("-inf")

# The evaluation reference never moves: K = 2 and replan 5, whatever teacher.yaml says.
REFERENCE_BEAM = 2
REFERENCE_REPLAN = 5
EVAL_FRAME_CAP = 6000          # SPEC §16
CHECK_FRAME_CAP = 3000
SLOW_BURST_V0, SLOW_BURST_V1 = 8.0, 16.0     # slow_burst_extra: full at L1 speed, 0 at L5 speed


@dataclass(frozen=True)
class TeacherConfig:
    beam: int = 2
    macro_frames: int = 5
    depth: int = 9
    clearance_weight: float = 0.5
    clearance_cap_m: float = 2.0
    climb_penalty: float = 0.05
    jump_extra_penalty: float = 0.95
    duck_extra_penalty: float = 0.95     # SPEC v1.7: the dive costs what the jump costs
    hold_gust: bool = True
    label_temperature: float = 2.0
    threat_frames: int = 60
    check_frame_cap: int = CHECK_FRAME_CAP
    safe_frames: int = 30               # safe[a]: some plan from a survives this many frames (1 s)
    descend_penalty: float = 0.0        # per DESCEND/DUCK macro (climb_penalty's mirror)
    slow_burst_extra: float = 0.0       # extra per JUMP/DUCK macro at 8 m/s, 0 at 16 m/s
    sensor_limited: bool = False        # plan only around the hazards the sensors show now
    keep_sustained_moves: bool = False  # never prune "keep climbing / keep descending"
    bird_envelope: bool = False         # plan birds as the band their sine path covers
    path_weight: float = 0.0            # + path_weight * mean over macros of min(path_time, H) / H
    path_horizon_s: float = 1.2         # H: how far ahead (in time) the path should stay clear
    path_margin_m: float = 0.3          # height margin around the hitbox for "the path is clear"
    parts_margin_m: float = 0.0         # plan parts-built obstacles grown by this much on every side
    tracked_movers: bool = False        # v2.3: plan movers as the tracker's predicted band
    tracked_fallback: bool = False      # v2.3: boxed in by the bands -> plan with the best-guess bands

    def to_dict(self) -> dict:
        return asdict(self)


OPTIONAL_DEFAULTS = {"descend_penalty": 0.0, "slow_burst_extra": 0.0, "sensor_limited": False,
                     "keep_sustained_moves": False, "bird_envelope": False, "path_weight": 0.0,
                     "path_horizon_s": 1.2, "path_margin_m": 0.3, "parts_margin_m": 0.0,
                     "tracked_movers": False, "tracked_fallback": False}


def load_teacher_config(path: str | Path | None = None, beam: int | None = None) -> TeacherConfig:
    """teacher.yaml -> TeacherConfig. ``beam`` (e.g. from --beam) overrides the file."""
    path = Path(path) if path is not None else TEACHER_YAML
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f for f in TeacherConfig.__dataclass_fields__}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown teacher settings {sorted(unknown)}")
    cfg = TeacherConfig(**raw)
    if beam is not None:
        cfg = replace(cfg, beam=int(beam))
    if cfg.beam < 1:
        raise ValueError("beam must be >= 1")
    return cfg


def reference_config(cfg: TeacherConfig | None = None) -> TeacherConfig:
    """The fixed evaluation reference: teacher.yaml's search and scoring, beam forced to 2."""
    cfg = cfg if cfg is not None else load_teacher_config()
    return replace(cfg, beam=REFERENCE_BEAM)


def reference_settings(cfg: TeacherConfig | None = None, frame_cap: int = EVAL_FRAME_CAP) -> dict:
    """What a planner cache entry records (and must match to be reused)."""
    ref = reference_config(cfg).to_dict()
    ref.pop("check_frame_cap", None)
    for k, v in OPTIONAL_DEFAULTS.items():      # options added later are recorded only when on, so
        if ref.get(k) == v:                     # existing cache entries stay valid
            ref.pop(k)
    ref["replan"] = REFERENCE_REPLAN
    ref["frame_cap"] = int(frame_cap)
    return ref


@dataclass
class Plan:
    scores: np.ndarray       # (5,) float32, -inf where masked
    safe: np.ndarray         # (5,) bool
    threat: int              # 0..4
    action: int


# ------------------------------------------------------------------------------------------
# search
# ------------------------------------------------------------------------------------------
def _run_macro(sim: Simulator, a: int, n: int) -> int:
    """Hold ``a`` for up to n frames (HOLD where masked). Returns frames survived."""
    avail = sim.available_actions
    step = sim.step
    for i in range(n):
        if step(a if a in avail() else HOLD):
            return i
    return n


def _search(root: Simulator, cfg: TeacherConfig) -> tuple[np.ndarray, np.ndarray]:
    """Beam search from a calm root: (scores[5], safe[5]) per available first action."""
    K = cfg.beam
    n = cfg.macro_frames
    depth = cfg.depth
    need = min(cfg.safe_frames, n * depth)
    w = cfg.clearance_weight / depth          # score += w * sum(clearance samples)
    cap = cfg.clearance_cap_m
    wp = cfg.path_weight / depth              # score += wp * sum(path samples), each min(path_time, H) / H
    ph, pm = cfg.path_horizon_s, cfg.path_margin_m

    def path(sim):
        return sim.path_time(ph, pm) / ph if wp else 0.0
    slow = min(max((SLOW_BURST_V1 - root.speed) / (SLOW_BURST_V1 - SLOW_BURST_V0), 0.0), 1.0)
    burst_extra = cfg.slow_burst_extra * slow
    pen_jump = cfg.climb_penalty + cfg.jump_extra_penalty + burst_extra
    pen_duck = cfg.descend_penalty + cfg.duck_extra_penalty + burst_extra
    pens = (cfg.climb_penalty, cfg.descend_penalty, pen_jump, pen_duck)      # ASCEND, DESCEND, JUMP, DUCK
    scores = np.full(C.N_ACTIONS, NEG_INF, dtype=np.float32)
    safe = np.zeros(C.N_ACTIONS, dtype=bool)

    for a in root.available_actions():
        c = root.clone(calm=True)             # clone of a calm sim: exact copy, stays calm
        f = _run_macro(c, a, n)
        pen = pens[a - 1] if a else 0.0
        if f < n:                             # crashed in the first macro
            scores[a] = f - pen
            safe[a] = f >= need
            continue
        cl = c.clearance()
        cs = cl if cl < cap else cap
        ps = path(c)
        best = NEG_INF
        longest = f                           # longest survival of any plan starting with a
        # beam node: (partial score, frames, clearance sum, path sum, penalty, sim, last macro's action)
        sustain = cfg.keep_sustained_moves
        beam = [(f + w * cs + wp * ps - pen, f, cs, ps, pen, c, a)]
        for _ in range(depth - 1):
            children = []
            for _sc, fr, csum, psum, pn, s, last in beam:
                for b in s.available_actions():
                    cc = s.clone(calm=True)
                    fb = _run_macro(cc, b, n)
                    pb = pn + (pens[b - 1] if b else 0.0)
                    if fb < n:                # crashed: finished plan, clearance 0 from here on
                        sc = fr + fb + w * csum + wp * psum - pb
                        if sc > best:
                            best = sc
                        if fr + fb > longest:
                            longest = fr + fb
                        continue
                    cl = cc.clearance()
                    cb = csum + (cl if cl < cap else cap)
                    pc = psum + path(cc)
                    children.append((fr + fb + w * cb + wp * pc - pb, fr + fb, cb, pc, pb, cc, b,
                                     sustain and b == last and (b == ASCEND or b == DESCEND)))
            if not children:
                break
            if children[0][1] > longest:
                longest = children[0][1]
            if len(children) > K:
                children.sort(key=lambda t: t[0], reverse=True)   # stable: action order on ties
                children = children[:K] + [ch for ch in children[K:] if ch[7]]
            beam = [ch[:7] for ch in children]
        else:
            # completed every depth: the beam holds full-horizon plans
            for sc, _fr, _cs, _ps, _pn, _s, _a in beam:
                if sc > best:
                    best = sc
        scores[a] = best
        safe[a] = longest >= need
    return scores, safe


def _threat(root: Simulator, cfg: TeacherConfig, safe: np.ndarray) -> int:
    """Frames until crash if the drone holds: 0 (> 2 s), 1 (1-2 s), 2 (0.5-1 s), 3 (< 0.5 s);
    4 if no action is safe."""
    if not safe.any():
        return 4
    h = root.clone(calm=True)
    tf = cfg.threat_frames
    for i in range(tf):
        if h.step(HOLD):
            crash_in = i + 1
            if crash_in > tf // 2:
                return 1
            if crash_in > tf // 4:
                return 2
            return 3
    return 0


def limit_to_sensors(c: Simulator, fog: bool) -> None:
    """Keep only the hazards the drone's sensors show now (as drone_env.sensors.observe selects
    them: leading edge within nose + range) and stop generation. Mutates the clone ``c``."""
    reach = c.distance_m + C.DRONE_X + C.HALF_W + (C.FOG_RANGE if fog else C.SENSOR_RANGE)
    t = c.frame * C.DT
    c.statics = [h for h in c.statics if h.x0 <= reach]
    c.pending = [h for h in c.pending if h.x0 <= reach]
    c.movers = [m for m in c.movers if m.xr - m.s * (t - m.tr) <= reach]
    c.dormant = []
    c.generate = False


def _level_box(m: Mover, lo: float, hi: float) -> Mover:
    """m flying level as a box from lo to hi (body edges)."""
    return m.copy(h=hi - lo, base=(lo + hi) / 2, amp=0.0, fm=0.0, dive=0.0)


def bird_envelopes(c: Simulator) -> None:
    """Replace every mover of the clone ``c`` that moves up and down (active and dormant: birds, v2.1
    traffic drift) by the band its path covers (from the game's own path settings)."""
    def env(m):
        if m.still:
            return m
        return _level_box(m, m.base - m.down - m.h / 2, m.base + m.up + m.h / 2)
    c.movers = [env(m) for m in c.movers]
    c.dormant = [env(m) for m in c.dormant]


def tracked_envelopes(c: Simulator, sim: Simulator, best_guess: bool = False) -> None:
    """Replace every mover of the clone ``c`` that ``sim``'s sensors show by the band the tracker
    predicts for it while the drone passes it (``drone_env.tracker.predict``: the pilot's v2.3
    hazard row; ``best_guess``: the band without margins). Use with ``sensor_limited`` (unseen
    movers are then gone)."""
    from drone_env import tracker
    pred = tracker.predict(sim)
    lo_i, hi_i = (6, 7) if best_guess else (3, 4)

    def env(m):
        p = pred.get(-m.uid if m.uid < 0 else m.uid)
        return m if p is None else _level_box(m, p[lo_i], p[hi_i])
    c.movers = [env(m) for m in c.movers]


def inflate_parts(c: Simulator, m: float) -> None:
    """Replace every parts-built obstacle of the clone ``c`` by a copy grown by ``m`` on every side
    (boxes and the hull outward, wires thicker). The original Static objects are left untouched."""
    grown = {}

    def grow(h):
        if not h.parts:
            return h
        g = grown.get(id(h))
        if g is None:
            parts = tuple((q[0], q[1] - m, q[2] - m, q[3] + m, q[4] + m) if q[0] == BOX else q[:5] + (q[5] + m,)
                          for q in h.parts)
            g = grown[id(h)] = Static(h.uid, h.kind, h.x0 - m, h.x1 + m, h.y0 - m, h.y1 + m, h.xc, h.r, parts)
        return g
    c.statics = [grow(h) for h in c.statics]
    c.pending = [grow(h) for h in c.pending]


def _root(sim: Simulator, cfg: TeacherConfig, best_guess: bool = False) -> Simulator:
    root = sim.clone(calm=True, hold_gust=cfg.hold_gust)
    if cfg.sensor_limited:
        limit_to_sensors(root, sim.fog_end > sim.frame)
    if cfg.tracked_movers:
        tracked_envelopes(root, sim, best_guess)
    elif cfg.bird_envelope:
        bird_envelopes(root)
    if cfg.parts_margin_m > 0.0:
        inflate_parts(root, cfg.parts_margin_m)
    return root


def plan(sim: Simulator, cfg: TeacherConfig) -> Plan:
    """Teacher scores, safety, threat and action for the current state of ``sim`` (not mutated)."""
    root = _root(sim, cfg)
    scores, safe = _search(root, cfg)
    if cfg.tracked_movers and cfg.tracked_fallback and not safe.any():
        root = _root(sim, cfg, best_guess=True)
        scores, safe = _search(root, cfg)
    threat = _threat(root, cfg, safe)
    return Plan(scores=scores, safe=safe, threat=threat, action=int(np.argmax(scores)))


def soft_targets(scores: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    """p_teacher = softmax(scores / T) over available actions (masked = -inf -> 0). Batched OK."""
    s = np.asarray(scores, dtype=np.float64) / temperature
    m = np.max(s, axis=-1, keepdims=True)
    e = np.where(np.isfinite(s), np.exp(s - m), 0.0)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


# ------------------------------------------------------------------------------------------
# flying the planner (check and evaluation reference)
# ------------------------------------------------------------------------------------------
def fly(level: str, seed: int, cfg: TeacherConfig, replan: int = 1, frame_cap: int = CHECK_FRAME_CAP,
        keep_times: bool = False) -> dict:
    """Fly the planner on a bare Simulator. replan=1 replans every frame; replan=5 commits to
    the first macro action (mask permitting) and replans every 5 frames."""
    sim = Simulator(level, seed)
    times = []
    pc = time.perf_counter
    while not sim.crashed and sim.frame < frame_cap:
        t0 = pc()
        a = plan(sim, cfg).action
        times.append((pc() - t0) * 1000.0)
        for _ in range(replan):
            sim.step(a if a in sim.available_actions() else HOLD)
            if sim.crashed or sim.frame >= frame_cap:
                break
    t = np.asarray(times) if times else np.zeros(1)
    out = {
        "level": level, "seed": int(seed), "distance_m": float(sim.distance_m), "frames": int(sim.frame),
        "crash_cause": sim.crash_cause, "survived": (not sim.crashed) and sim.frame >= frame_cap,
        "decisions": len(times), "decision_ms_mean": float(t.mean()),
        "decision_ms_p50": float(np.percentile(t, 50)), "decision_ms_p95": float(np.percentile(t, 95)),
        "decision_ms_p99": float(np.percentile(t, 99)),
    }
    if keep_times:
        out["times_ms"] = times
    return out


def _fly_task(task: tuple) -> dict:
    level, seed, cfg_dict, replan, cap, keep = task
    return fly(level, seed, TeacherConfig(**cfg_dict), replan, cap, keep_times=keep)


def fly_many(level: str, seeds, cfg: TeacherConfig, replan: int, frame_cap: int, workers: int = 6,
             keep_times: bool = False, progress: bool = True):
    """Fly the planner on several seeds in parallel. Yields result dicts as they finish."""
    from teacher.parallel import parallel_map
    tasks = [(level, int(s), cfg.to_dict(), replan, frame_cap, keep_times) for s in seeds]
    for _i, r in parallel_map(_fly_task, tasks, workers):
        if progress:
            print(f"  {r['level']} seed {r['seed']}: {r['distance_m']:7.1f} m  {r['frames']:5d} frames  "
                  f"{r['crash_cause'] or 'survived'}  {r['decision_ms_mean']:.2f} ms/decision", flush=True)
        yield r


# ------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------
def check(level: str, split: str, beam: int | None, workers: int, max_seeds: int | None,
          frame_cap: int | None = None, profile: bool = False, config: str | None = None,
          skip_seeds: int = 0) -> dict:
    from drone_env.seeds import seeds_for
    if split not in ("val", "train"):
        raise SystemExit("--check only runs on VAL (or TRAIN) seeds; TEST and STRESS are for final evaluation")
    cfg = load_teacher_config(config, beam=beam)
    cap = frame_cap or cfg.check_frame_cap
    seeds = seeds_for(split, level)[skip_seeds:][:max_seeds]
    print(f"planner --check  level={level} seeds={split} x{len(seeds)}  beam={cfg.beam}  replan=1  "
          f"cap={cap} frames  workers={workers}  hold_gust={cfg.hold_gust}  "
          f"jump_extra_penalty={cfg.jump_extra_penalty}  duck_extra_penalty={cfg.duck_extra_penalty}", flush=True)
    t0 = time.time()
    if profile:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        res = [fly(level, s, cfg, 1, cap, keep_times=True) for s in seeds]
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(12)
    else:
        res = list(fly_many(level, seeds, cfg, 1, cap, workers, keep_times=True))
    wall = time.time() - t0
    times = np.concatenate([np.asarray(r["times_ms"]) for r in res])
    surv = sum(r["survived"] for r in res)
    dist = np.array([r["distance_m"] for r in res])
    causes = collections.Counter(r["crash_cause"] or "survived" for r in res)
    share = surv / len(res)
    print(f"\nsurvived {cap} frames: {surv}/{len(res)} = {100 * share:.1f}%   (G2 needs >= 90%)")
    print(f"mean distance: {dist.mean():.1f} m   median {np.median(dist):.1f} m")
    print(f"outcomes: {dict(causes)}")
    label = "G2 timing (workers=1)" if workers <= 1 else f"contended timing (workers={workers}; use --workers 1 for G2)"
    print(f"ms per decision: mean {times.mean():.2f}  p95 {np.percentile(times, 95):.2f}  "
          f"over {len(times):,} decisions   [{label}; G2 needs mean <= 60 ms]")
    print(f"wall time {wall:.0f} s")
    return {"level": level, "split": split, "beam": cfg.beam, "workers": workers, "seeds": len(res),
            "survived": surv, "survival_share": share, "mean_distance_m": float(dist.mean()),
            "ms_mean": float(times.mean()), "ms_p95": float(np.percentile(times, 95)),
            "decisions": int(len(times)), "outcomes": dict(causes), "wall_s": wall}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m teacher.planner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fly the planner (replan 1) and report G2 numbers")
    ap.add_argument("--level", default="L5", choices=[l for l in C.LEVELS if l != "stress"])
    ap.add_argument("--seeds", default="val", choices=["val", "train"])
    ap.add_argument("--beam", type=int, default=None, help="override beam K from teacher.yaml")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-seeds", type=int, default=None)
    ap.add_argument("--skip-seeds", type=int, default=0, help="skip the first N seeds of the split")
    ap.add_argument("--frame-cap", type=int, default=None, help="default: check_frame_cap (3000)")
    ap.add_argument("--profile", action="store_true", help="cProfile one process (implies --workers 1)")
    ap.add_argument("--out", default=None, help="optional JSON file for the summary")
    ap.add_argument("--config", default=None, help="teacher settings file (default configs/train/teacher.yaml)")
    args = ap.parse_args(argv)
    if not args.check:
        ap.print_help()
        return 2
    s = check(args.level, args.seeds, args.beam, 1 if args.profile else args.workers, args.max_seeds,
              args.frame_cap, args.profile, args.config, args.skip_seeds)
    if args.out:
        import json
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
