"""Planes and the JUMP/DUCK reflexes (SPEC v1.7). Planes are retired in v2.0 (weight 0 in every
level), so only the placed-plane mechanics are tested; the tests of generated planes were removed."""
import math

import numpy as np
import pytest

from drone_env import constants as C
from drone_env import events
from drone_env.generator import Mover, Static, footprint_of, plane_escape_ok, visible_x
from drone_env.sim import Simulator

from .helpers import empty_sim

REFLEXES = [(C.HOLD, 0, C.HOLD), (C.ASCEND, 0, C.ASCEND), (C.DESCEND, 0, C.DESCEND)] + [
    (a, w, b) for w in C.ESCAPE_WAITS for a, b in
    ((C.JUMP, C.HOLD), (C.JUMP, C.ASCEND), (C.DUCK, C.HOLD), (C.DUCK, C.DESCEND))]


def fly_reflex(sim, vy0, plan, x_stop):
    """Real Simulator: keep the current action while reacting, then the reflex. True if the
    drone is still flying once its centre is past x_stop."""
    keep = C.ASCEND if vy0 > 0.5 else (C.DESCEND if vy0 < -0.5 else C.HOLD)
    first, wait, follow = plan
    burst_like = first in (C.JUMP, C.DUCK)
    j = 0
    while sim.x <= x_stop:
        if j < C.REACTION_FRAMES:
            a = keep
        elif j < C.REACTION_FRAMES + wait:
            a = C.HOLD
        elif j == C.REACTION_FRAMES + wait or not burst_like:
            a = first
        else:
            a = follow
        if sim.step(a if a in sim.available_actions() else C.HOLD):
            return False
        j += 1
    return True


def plane_scene(level, y0, vy0, plane_y, s_plane=8.0):
    sim = empty_sim(y0, level)
    events.make_calm(sim)
    sim.vy = vy0
    sim.place_mover(C.PLANE, sim.x + C.HALF_W + C.SENSOR_RANGE, plane_y, 5.0, 1.0, s=s_plane)
    return sim


def test_plane_hits_like_any_hazard():
    sim = plane_scene("L5", 5.0, 0.0, 5.0)
    for _ in range(60):
        if sim.step(C.HOLD):
            break
    assert sim.crashed and sim.crash_cause == "plane"
    o = plane_scene("L5", 5.0, 0.0, 5.0).observe()
    assert o["hazard_type"][0] == C.PLANE and o["hazards"][0, 7] == 1.0
    assert o["hazards"][0, 4] == pytest.approx(-(12.0 + 8.0) / 16)     # closing at 20 m/s (L5 12 m/s, v2.3)


@pytest.mark.parametrize("vy0", [-1.5, 1.5])
def test_at_l5_only_a_burst_escapes_a_plane_seen_late(vy0):
    """A plane at the drone's height, first seen 16 m ahead, closing at 24 m/s: while sinking,
    ASCEND cannot turn the drone around in time (and DESCEND cannot while climbing); JUMP and
    DUCK set the vertical speed at once."""
    x_stop = None

    def run(plan):
        sim = plane_scene("L5", 5.0, vy0, 5.0)
        m = sim.movers[0]
        return fly_reflex(sim, vy0, plan, m.xr + m.w + 1.0)

    slow = (C.ASCEND, 0, C.ASCEND) if vy0 < 0 else (C.DESCEND, 0, C.DESCEND)
    assert not run((C.HOLD, 0, C.HOLD)) and not run(slow)
    assert run((C.JUMP, 0, C.HOLD)) and run((C.DUCK, 0, C.HOLD))


def test_escape_check_refuses_a_plane_with_nowhere_to_go():
    """Building below and a storm dip above leave only the plane's own band: no reflex works."""
    sched = Simulator("L5", 0, generate=False).schedule
    xe = 40.0
    plane = Mover(1, C.PLANE, 0, 5.0, 1.0, 5.0, 0.0, 0.0, 0.0, 8.0, xe)
    b = Static(2, C.BUILDING, xe - 6.0, xe + 10.0, 0.0, 3.6)
    st = Static(3, C.STORM, xe - 8.0, xe + 12.0, 6.6, C.CEILING, r=0.5)
    fps = [footprint_of(h) for h in (plane, b, st)]
    xv = visible_x(plane, 16.0)
    x_stop = footprint_of(plane).fx1
    k_end = int(math.ceil(max(fp.fx1 for fp in fps) / C.CHECK_DX)) + 1
    assert not plane_escape_ok([(4.5, 5.5)], xv, x_stop, fps, sched, k_end)
    # open sky: the same plane is escapable from anywhere around its height
    fps2 = [footprint_of(plane)]
    k2 = int(math.ceil(fps2[0].fx1 / C.CHECK_DX)) + 1
    assert plane_escape_ok([(4.0, 6.0)], xv, x_stop, fps2, sched, k2)
