"""Optional teacher settings, tested against the plain search (TeacherConfig(): all options off); teacher.yaml (arc's
teacher) turns several on."""
from dataclasses import replace

import numpy as np
import pytest

from drone_env import constants as C
from teacher.planner import OPTIONAL_DEFAULTS, TeacherConfig, load_teacher_config, plan, reference_settings
from tests.env.helpers import empty_sim


@pytest.fixture(scope="module")
def cfg():
    return TeacherConfig()


def test_options_are_recorded_in_the_reference_only_when_on(cfg):
    for k, v in OPTIONAL_DEFAULTS.items():
        assert getattr(cfg, k) == v
        assert k not in reference_settings(cfg)
    assert reference_settings(replace(cfg, sensor_limited=True))["sensor_limited"] is True
    arc = load_teacher_config()                  # teacher.yaml: arc's teacher, its search = the plain search's
    on = {k for k, v in OPTIONAL_DEFAULTS.items() if getattr(arc, k) != v}
    assert on == {"descend_penalty", "slow_burst_extra", "sensor_limited", "keep_sustained_moves", "parts_margin_m",
                  "tracked_movers", "tracked_fallback"}
    assert set(reference_settings(arc)) - set(reference_settings(cfg)) == on
    assert replace(arc, **{k: OPTIONAL_DEFAULTS[k] for k in on}) == cfg


def test_descend_penalty_costs_one_macro_in_open_sky(cfg):
    """At 7 m every plan keeps the capped 2 m clearance except climbs, so the best DESCEND and
    DUCK plans hold after the first macro and pay the penalty exactly once."""
    base = plan(empty_sim(7.0, "L1"), cfg).scores
    pen = plan(empty_sim(7.0, "L1"), replace(cfg, descend_penalty=0.05)).scores
    assert pen[C.DESCEND] == pytest.approx(base[C.DESCEND] - 0.05)
    assert pen[C.DUCK] == pytest.approx(base[C.DUCK] - 0.05)
    assert pen[C.HOLD] == base[C.HOLD] and pen[C.ASCEND] == base[C.ASCEND] and pen[C.JUMP] == base[C.JUMP]


@pytest.mark.parametrize("level, extra", [("L1", 2.25), ("L3", 1.5), ("L5", 1.5)])   # 10 / 12 / 12 m/s (v2.3)
def test_slow_burst_extra_fades_from_8_to_16_mps(cfg, level, extra):
    base = plan(empty_sim(5.0, level), cfg).scores
    slow = plan(empty_sim(5.0, level), replace(cfg, slow_burst_extra=3.0)).scores
    assert slow[C.JUMP] == pytest.approx(base[C.JUMP] - extra, abs=1e-5)
    assert slow[C.DUCK] == pytest.approx(base[C.DUCK] - extra, abs=1e-5)
    assert np.array_equal(slow[:C.JUMP], base[:C.JUMP])


@pytest.mark.parametrize("ahead, seen", [(12.0, False), (6.0, True)])
def test_sensor_limited_ignores_what_the_sensors_cannot_see(cfg, ahead, seen):
    """In fog (8 m sensor range): a wall (ground to ceiling) 12 m ahead at 12 m/s is reached within
    the 1.5 s plan but lies beyond what the sensors show; at 6 m both views see it."""
    from drone_env import events

    def scene():
        s = empty_sim(5.0, "L5")
        events.start(s, events.FOG, 5.0)
        x0 = s.distance_m + C.DRONE_X + C.HALF_W + ahead
        s.place_static(C.BUILDING, x0, x0 + 2.0, 0.0, C.CEILING)
        return s
    full = plan(scene(), cfg).scores.max()
    sim = scene()
    n = len(sim.statics)
    lim = plan(sim, replace(cfg, sensor_limited=True)).scores.max()
    assert len(sim.statics) == n and sim.generate is False      # the sim itself is untouched
    assert full < 45
    assert (lim < 45) if seen else (lim >= 45)



@pytest.mark.parametrize("ahead", [10.0, 13.0])
def test_keep_sustained_moves_finds_the_steady_climb(ahead):
    """A building 0-5 m tall, 10 or 13 m ahead at L1 (1.0 or 1.3 s at 10 m/s), drone at 3 m: ASCEND gets over it
    (3 m/s), but pruning by partial score drops every steady climb, so only the JUMP-first plan
    survived. Keeping "keep climbing" finds it, and the teacher climbs instead of jumping."""
    cfg = TeacherConfig(sensor_limited=True, descend_penalty=0.05)

    def scene():
        s = empty_sim(3.0, "L1")
        x0 = s.distance_m + C.DRONE_X + C.HALF_W + ahead
        s.place_static(C.BUILDING, x0, x0 + 2.0, 0.0, 5.0)
        return s
    old = plan(scene(), cfg)
    new = plan(scene(), replace(cfg, keep_sustained_moves=True))
    assert old.action == C.JUMP and old.scores[C.ASCEND] < 45
    assert new.action == C.ASCEND and new.scores[C.ASCEND] >= 45 and new.safe[C.ASCEND]
    assert np.array_equal(plan(empty_sim(5.0, "L1"), replace(cfg, keep_sustained_moves=True)).scores,
                          plan(empty_sim(5.0, "L1"), cfg).scores)


def test_bird_envelope_plans_the_whole_band():
    """A bird bobbing +-1.5 m around the drone's height, 6 m ahead at L1: its exact path can leave a
    gap at the moment it passes; as a band (5 m centre +- 1.65 m) it can only be passed over or under."""
    from drone_env.sim import Simulator
    sim = empty_sim(5.0, "L1")
    sim.place_mover(C.BIRD, sim.x + C.HALF_W + 6.0, 5.0, 0.5, 0.3, s=2.0, amp=1.5, period=1.5)
    cfg = TeacherConfig(sensor_limited=True)
    n = len(sim.movers)
    env = plan(sim, replace(cfg, bird_envelope=True))
    assert len(sim.movers) == n and sim.movers[0].amp == 1.5          # the sim itself is untouched
    assert not env.safe[C.HOLD]                                        # the band is at the drone's height
    c = sim.clone(calm=True)
    from teacher.planner import bird_envelopes
    bird_envelopes(c)
    m = c.movers[0]
    assert m.amp == 0.0 and m.h == pytest.approx(0.3 + 3.0) and m.base == 5.0


@pytest.mark.parametrize("ahead", [16.5, 19.0])
def test_path_weight_starts_the_climb_earlier(ahead):
    """A building 0-5 m tall 16.5 or 19 m ahead at L1 (1.65 or 1.9 s at 10 m/s), drone at 3 m: no crash fits in the
    1.5 s plan yet, so the v20 teacher holds; a bonus for a clear level path ahead starts the climb now."""
    cfg = TeacherConfig(sensor_limited=True, descend_penalty=0.05, slow_burst_extra=3.0,
                        keep_sustained_moves=True, bird_envelope=True)

    def scene():
        s = empty_sim(3.0, "L1")
        x0 = s.distance_m + C.DRONE_X + C.HALF_W + ahead
        s.place_static(C.BUILDING, x0, x0 + 2.0, 0.0, 5.0)
        return s
    assert plan(scene(), cfg).action == C.HOLD
    assert plan(scene(), replace(cfg, path_weight=2.0)).action == C.ASCEND
    open_sky = plan(empty_sim(5.0, "L1"), replace(cfg, path_weight=2.0))
    assert open_sky.action == C.HOLD


def test_parts_margin_keeps_distance_from_wires_and_leaves_the_sim_alone():
    """A power line ahead at the drone's height: grown by 0.3 m it covers 0.3 m more on every side, the
    clone's copies change and the sim's own obstacle does not; with the margin the teacher still finds
    a way past, and an obstacle without parts (a building) is not grown."""
    from teacher.planner import inflate_parts
    sim = empty_sim(5.0, "L1")
    nose = sim.distance_m + C.DRONE_X + C.HALF_W
    h = sim.gen._sample_parts(C.POWERLINE, nose + 6.0, np.random.default_rng(2))
    h.uid = 999
    b = sim.place_static(C.BUILDING, nose + 40.0, nose + 42.0, 0.0, 3.0)
    sim.statics = sorted([h, b], key=lambda s: s.x0)
    sim.pending = list(sim.statics)
    before = (h.x0, h.x1, h.y0, h.y1, h.parts)
    c = sim.clone(calm=True)
    inflate_parts(c, 0.3)
    g = next(s for s in c.statics if s.uid == 999)
    assert (h.x0, h.x1, h.y0, h.y1, h.parts) == before and g is not h
    assert g.x0 == pytest.approx(h.x0 - 0.3) and g.y1 == pytest.approx(h.y1 + 0.3)
    assert all(q[5] == pytest.approx(p[5] + 0.3) for p, q in zip(h.parts, g.parts))
    assert next(s for s in c.statics if s.kind == C.BUILDING) is b and c.pending[0] is g
    cfg = TeacherConfig(sensor_limited=True, keep_sustained_moves=True)
    for y in (0.5 * (h.y0 + h.y1), h.y1 + C.HALF_H + 0.1):
        sim.y = y
        p = plan(sim, replace(cfg, parts_margin_m=0.3))
        assert p.safe.any()
    sim.y = h.y1 + C.HALF_H + 0.1          # 0.1 m over the top wire: flying level is fine exactly, not with a 0.3 m margin
    for margin, crash in ((0.0, False), (0.3, True)):
        c = sim.clone(calm=True)
        inflate_parts(c, margin)
        while not c.crashed and c.distance_m + C.DRONE_X - C.HALF_W < h.x1:
            c.step(C.HOLD)
        assert c.crashed == crash
