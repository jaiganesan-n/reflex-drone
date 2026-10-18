import time

import numpy as np

from drone_env import constants as C
from drone_env import events
from drone_env.generator import NEVER
from drone_env.sim import Simulator

from .helpers import scripted


def advance(s, n, start=0):
    for f in range(start, start + n):
        s.step(scripted(s, f))


def test_clone_is_independent_and_does_not_perturb_the_original():
    a = Simulator("L5", 100_003)
    twin = Simulator("L5", 100_003)
    advance(a, 200)
    advance(twin, 200)
    h = a.state_hash()
    for calm in (True, False):
        c = a.clone(calm=calm)
        advance(c, 400, 200)            # forces generation and RNG draws in the clone
        c.observe()
        assert a.state_hash() == h
    # the original's future is unchanged by having been cloned
    advance(a, 600, 200)
    advance(twin, 600, 200)
    assert a.state_hash() == twin.state_hash()


def test_non_calm_clone_reproduces_the_original():
    a = Simulator("stress", 300_005)
    advance(a, 150)
    c = a.clone(calm=False)
    advance(a, 500, 150)
    advance(c, 500, 150)
    assert a.state_hash() == c.state_hash()


def _state_with_gust_and_surge():
    for seed in range(300_000, 300_050):
        s = Simulator("stress", seed)
        for f in range(1500):
            if s.gust_active and s.surge_active:
                return s
            s.step(scripted(s, f))
            if s.crashed:
                break
    raise AssertionError("no state with gust + surge found")


def test_calm_clone_removes_gust_and_surge_and_stops_new_interruptions():
    s = _state_with_gust_and_surge()
    c = s.clone(calm=True)
    assert not c.gust_active and not c.surge_active
    assert c.speed == c.schedule.speed_at(c.distance_m)
    fog_end = c.fog_end
    for f in range(900):
        c.step(scripted(c, f))
        assert c.gust_a == 0.0 and not c.surge_active and c.fog_end == fog_end
        assert c.next_gust == c.next_surge == c.next_fog == NEVER
    # the original keeps its interruptions
    assert s.gust_active and s.surge_active


def test_hold_gust_option_keeps_the_felt_gust():
    s = _state_with_gust_and_surge()
    g = s.gust_a
    c = s.clone(calm=True, hold_gust=True)
    assert c.gust_a == g and not c.surge_active
    advance(c, 200)
    assert c.gust_a == g
    # a clone of a calm clone is an exact (still calm) copy
    cc = c.clone(calm=True)
    assert cc.gust_a == g and cc.calm


def test_planner_api():
    s = Simulator("L3", 100_000)
    advance(s, 100)
    c = s.clone()
    assert isinstance(c.step(C.HOLD), bool)
    assert c.frame == s.frame + 1 and c.distance_m > s.distance_m
    assert c.crashed is False and c.crash_cause is None
    m = c.action_mask()
    assert m.dtype == np.int8 and m.shape == (5,)
    assert tuple(np.flatnonzero(m)) == c.available_actions()
    assert c.clearance() >= 0.0
    o = c.observe()
    assert set(o) == {"self", "rays", "hazards", "hazard_type", "hazard_mask", "action_mask"}


def test_clone_speed():
    s = Simulator("L5", 100_001)
    advance(s, 300)
    n = 5000
    t0 = time.perf_counter()
    for _ in range(n):
        s.clone(calm=True)
    us = (time.perf_counter() - t0) / n * 1e6
    assert us <= 50.0, f"clone took {us:.1f} us"
