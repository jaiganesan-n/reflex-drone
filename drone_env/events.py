"""Random interruptions: gusts, speed surges and fog banks (SPEC §3).

Arrivals are a Poisson process per type: the next arrival frame is drawn from an exponential
with the level's mean interval when the previous one fires, so a frame without an arrival costs
three integer comparisons and no RNG draw. Draws come from the simulator's *events* stream of the
world RNG, never from the sensor RNG.

All state lives on the Simulator (plain attributes), which keeps clone() a dict copy.
"""
from __future__ import annotations

import math

from . import constants as C
from .generator import NEVER

GUST, SURGE, FOG = 0, 1, 2


def _rng(sim):
    sim._rng_events = sim._rng_events.own()
    return sim._rng_events.rng


def _interval_frames(rng, mean_s: float) -> int:
    return max(1, int(math.ceil(rng.exponential(mean_s) * C.FPS)))


def reschedule(sim) -> None:
    """(Re)draw the next arrival of every interruption type from the current level's rates."""
    sim._ev_idx = sim.schedule.index_at(sim.distance_m)
    if sim.calm:
        sim.next_gust = sim.next_surge = sim.next_fog = NEVER
        return
    spec = sim.schedule.specs[sim._ev_idx]
    f = sim.frame
    if spec.gust_every > 0 or spec.surge_every > 0 or spec.fog_every > 0:
        rng = _rng(sim)
    sim.next_gust = f + _interval_frames(rng, spec.gust_every) if spec.gust_every > 0 else NEVER
    sim.next_surge = f + _interval_frames(rng, spec.surge_every) if spec.surge_every > 0 else NEVER
    sim.next_fog = f + _interval_frames(rng, spec.fog_every) if spec.fog_every > 0 else NEVER


def any_active(sim) -> bool:
    f = sim.frame
    return sim.gust_a != 0.0 or sim.surge_end > f or sim.fog_end > f


def start(sim, kind: int, duration_s: float | None = None, accel: float | None = None) -> None:
    """Start an interruption now. Also used by tests to force one."""
    f = sim.frame
    if kind == GUST:
        if accel is None or duration_s is None:
            rng = _rng(sim)
            spec = sim.schedule.specs[sim._ev_idx]
            mag = rng.uniform(spec.gust_lo, spec.gust_hi)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            accel = sign * mag if accel is None else accel
            duration_s = rng.uniform(*C.GUST_S) if duration_s is None else duration_s
        sim.gust_a = float(accel)
        sim.gust_end = f + max(1, int(round(duration_s * C.FPS)))
    elif kind == SURGE:
        frames = C.SURGE_FRAMES if duration_s is None else max(1, int(round(duration_s * C.FPS)))
        sim.surge_end = max(sim.surge_end, f + frames)
    elif kind == FOG:
        if duration_s is None:
            duration_s = _rng(sim).uniform(*C.FOG_S)
        sim.fog_end = max(sim.fog_end, f + max(1, int(round(duration_s * C.FPS))))
    else:
        raise ValueError(kind)


def update(sim) -> None:
    """Called once per frame after the frame counter advanced."""
    f = sim.frame
    if sim.gust_a != 0.0 and f >= sim.gust_end:
        sim.gust_a = 0.0
    if sim.calm:
        return
    if sim.schedule.multi and sim.schedule.index_at(sim.distance_m) != sim._ev_idx:
        reschedule(sim)
    if f < sim.next_gust and f < sim.next_surge and f < sim.next_fog:
        return
    spec = sim.schedule.specs[sim._ev_idx]
    for kind in (GUST, SURGE, FOG):
        nxt = (sim.next_gust, sim.next_surge, sim.next_fog)[kind]
        if f < nxt:
            continue
        blocked = spec.exclusive and any_active(sim)
        if not blocked:
            start(sim, kind)
        rng = _rng(sim)
        mean = (spec.gust_every, spec.surge_every, spec.fog_every)[kind]
        n = f + _interval_frames(rng, mean)
        if kind == GUST:
            sim.next_gust = n
        elif kind == SURGE:
            sim.next_surge = n
        else:
            sim.next_fog = n


def make_calm(sim) -> None:
    """Remove the active gust and surge and stop new random interruptions (planner clones)."""
    sim.calm = True
    sim.gust_a = 0.0
    sim.gust_end = 0
    sim.surge_end = 0
    sim.next_gust = sim.next_surge = sim.next_fog = NEVER
