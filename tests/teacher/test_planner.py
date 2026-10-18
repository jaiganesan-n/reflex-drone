"""Teacher planner: determinism, purity, label semantics, and that it out-flies the rule agent."""
import numpy as np
import pytest

from app.rule_eval import fly as fly_rule
from drone_env.sim import Simulator
from teacher.planner import (REFERENCE_BEAM, TeacherConfig, fly, load_teacher_config, plan, reference_config,
                             reference_settings, soft_targets)

CAP = 600
L3_SEEDS = (100002, 100009, 100011)     # L3 VAL seeds where `rule` crashes within 600 frames (SPEC v2.3)


@pytest.fixture(scope="module")
def cfg():
    return load_teacher_config()


def _advance(sim, n, a=0):
    for _ in range(n):
        sim.step(a)
    return sim


def test_teacher_yaml_loads_and_reference_is_fixed(cfg, tmp_path):
    assert cfg.beam >= 1 and cfg.macro_frames == 5
    assert reference_config(TeacherConfig(beam=1)).beam == REFERENCE_BEAM == 2
    ref = reference_settings(cfg, 6000)
    assert ref["replan"] == 5 and ref["beam"] == 2 and ref["frame_cap"] == 6000
    bad = tmp_path / "t.yaml"
    bad.write_text("beam: 2\nnot_a_setting: 1\n")
    with pytest.raises(ValueError):
        load_teacher_config(bad)
    assert load_teacher_config(beam=1).beam == 1


def test_plan_is_deterministic_and_does_not_touch_the_sim(cfg):
    sim = _advance(Simulator("L5", 100_001), 200)
    h0 = sim.state_hash()
    p1, p2 = plan(sim, cfg), plan(sim, cfg)
    assert sim.state_hash() == h0
    assert np.array_equal(p1.scores, p2.scores) and np.array_equal(p1.safe, p2.safe)
    assert (p1.threat, p1.action) == (p2.threat, p2.action)
    # same state reached twice gives the same plan
    p3 = plan(_advance(Simulator("L5", 100_001), 200), cfg)
    assert np.array_equal(p1.scores, p3.scores) and p1.action == p3.action


def test_fly_is_deterministic(cfg):
    a = fly("L3", 100_002, cfg, replan=1, frame_cap=300)
    b = fly("L3", 100_002, cfg, replan=1, frame_cap=300)
    for k in ("distance_m", "frames", "crash_cause", "decisions"):
        assert a[k] == b[k]


def test_label_semantics(cfg):
    sim = Simulator("L3", 100_000)
    sim.step(3)                                      # JUMP: burst -> only HOLD available
    p = plan(sim, cfg)
    assert np.isfinite(p.scores[0]) and np.all(np.isneginf(p.scores[1:]))
    assert not p.safe[1:].any() and p.action == 0
    assert 0 <= p.threat <= 4
    pt = soft_targets(p.scores, cfg.label_temperature)
    assert pt[0] == pytest.approx(1.0) and np.all(pt[1:] == 0)
    # scores are frames survived (<= depth * macro) plus a small clearance term
    sim2 = _advance(Simulator("L1", 100_000), 30)
    p2 = plan(sim2, cfg)
    horizon = cfg.depth * cfg.macro_frames
    assert np.all(p2.scores <= horizon + cfg.clearance_weight * cfg.clearance_cap_m + 1e-5)
    assert p2.safe.all() and p2.threat == 0          # open sky at the start of L1
    assert p2.action == int(np.argmax(p2.scores))


def test_threat_four_when_nothing_is_safe(cfg):
    sim = Simulator("L1", 100_000)
    sim.y, sim.vy = 9.0, 6.0                        # shooting into the ceiling
    sim.cooldown = 20                               # no JUMP/DUCK available
    p = plan(sim, cfg)
    assert not p.safe.any() and p.threat == 4


def test_only_the_dive_escapes_a_ceiling_crash(cfg):
    """SPEC v1.7: DUCK sets vy = -6 at once, so it is the one reflex that saves this state."""
    sim = Simulator("L1", 100_000)
    sim.y, sim.vy = 9.3, 6.0                        # braking at 12 m/s^2 needs 1.5 m: too late
    p = plan(sim, cfg)
    assert p.safe.tolist() == [False, False, False, False, True] and p.action == 4


def test_planner_beats_rule_on_l3(cfg):
    for s in L3_SEEDS:
        r = fly_rule("rule", "L3", s, CAP)
        p = fly("L3", s, cfg, replan=1, frame_cap=CAP)
        assert r["crash_cause"] is not None
        assert p["distance_m"] > r["distance_m"], (s, p, r)
