import pytest

from drone_env import constants as C
from drone_env import events
from drone_env.sim import Simulator

from .helpers import empty_sim, reference_climb


@pytest.mark.parametrize("seq", [
    [C.HOLD] * 60,
    [C.ASCEND] * 30,
    [C.DESCEND] * 30,
    [C.JUMP] + [C.HOLD] * 40,
    [C.DUCK] + [C.HOLD] * 20,
    [C.DUCK, C.DESCEND, C.DESCEND] + [C.HOLD] * 12,
    [C.ASCEND] * 10 + [C.HOLD] * 10 + [C.DESCEND] * 10 + [C.JUMP] * 35,
])
def test_vertical_motion_matches_spec_equations(seq):
    s = empty_sim(5.0)
    for a in seq:
        s.step(a)
    y, vy = reference_climb(seq, 5.0)
    assert s.y == pytest.approx(y, abs=1e-12) and s.vy == pytest.approx(vy, abs=1e-12)


def test_directions_and_amounts():
    s = empty_sim(5.0)
    for _ in range(60):
        s.step(C.HOLD)
    assert s.y == pytest.approx(5.0)
    s = empty_sim(5.0)
    for _ in range(30):
        s.step(C.ASCEND)
    assert 2.4 < s.y - 5.0 < 3.0 and s.vy == pytest.approx(3.0, abs=0.01)
    s = empty_sim(5.0)
    for _ in range(30):
        s.step(C.DESCEND)
    assert 2.4 < 5.0 - s.y < 3.0
    s = empty_sim(2.0)
    s.step(C.JUMP)
    assert s.vy == pytest.approx(6.0)
    for _ in range(8):
        s.step(C.HOLD)
    assert s.y - 2.0 == pytest.approx(9 * 6.0 * C.DT)       # the burst holds 6 m/s for 9 frames
    s = empty_sim(5.0)
    s.step(C.DUCK)
    assert s.half_h == pytest.approx(C.HALF_H * 0.6) and s.v_target == C.DUCK_V
    assert s.vy == pytest.approx(-6.0)
    for _ in range(8):
        s.step(C.HOLD)
    assert 5.0 - s.y == pytest.approx(9 * 6.0 * C.DT)       # the dive holds -6 m/s for 9 frames
    s.step(C.HOLD)
    assert s.half_h == pytest.approx(C.HALF_H)              # untucked after the dive


def test_wind_gust_and_downdraft():
    s = empty_sim(5.0)
    events.start(s, events.GUST, duration_s=2.0, accel=6.0)
    for _ in range(45):
        s.step(C.HOLD)
    assert s.vy == pytest.approx(6.0 / 8.0, abs=0.01)         # HOLD equilibrium against the gust
    assert s.observe()["self"][5] == pytest.approx(6.0 / 8.0)
    for _ in range(30):
        s.step(C.HOLD)
    assert not s.gust_active and s.observe()["self"][5] == 0.0
    s = empty_sim(5.0)
    s.place_static(C.DOWNDRAFT, s.x - 2, s.x + 30, 0, 10)
    assert s.a_wind == C.DOWNDRAFT_ACCEL
    for _ in range(45):
        s.step(C.HOLD)
    assert s.vy == pytest.approx(-3.0 / 8.0, abs=0.01)


def test_scroll_speed():
    s = Simulator("L1", 0, generate=False)
    for _ in range(30):
        s.step(C.HOLD)
    assert s.distance_m == pytest.approx(10.0)
    # SPEC v2.3: L1 10 m/s, L2-L5 12 m/s
    for level, v in [("L1", 10.0), ("L2", 12.0), ("L3", 12.0), ("L4", 12.0), ("L5", 12.0), ("stress", 12.0 * 1.2)]:
        s = Simulator(level, 0, generate=False)
        assert s.speed == pytest.approx(v)
        s.distance_m = 5000.0                       # fixed levels keep their speed
        s._update_local()
        assert s.speed == pytest.approx(v) and s.level_now == level
    s = Simulator("L5", 0, generate=False)
    events.start(s, events.SURGE)
    s._update_local()
    assert s.speed == pytest.approx(12.0 * 1.3)
    s = Simulator("endless", 0, generate=False)
    # v2.3: 10 -> 12 m/s over L1, then 12 m/s; levels every 1,750 m (L5+ gets denser instead of faster)
    for d, lvl, v in [(0, "L1", 10.0), (875, "L1", 11.0), (1749, "L1", 12.0 - 2 / 1750), (1750, "L2", 12.0),
                      (3500, "L3", 12.0), (5250, "L4", 12.0), (6999, "L4", 12.0),
                      (7000, "L5", 12.0), (10_000, "L5", 12.0), (15_000, "L5", 12.0), (40_000, "L5", 12.0)]:
        s.distance_m = float(d)
        s._update_local()
        assert s.level_now == lvl and s.speed == pytest.approx(v)


def test_gust_strength_follows_the_level():
    for level, (lo, hi) in [("L1", (2.0, 4.0)), ("L3", (4.0, 6.0)), ("L5", (4.0, 8.0))]:
        mags = []
        for seed in range(40):
            s = Simulator(level, seed, generate=False)
            events.start(s, events.GUST)
            mags.append(abs(s.gust_a))
        assert lo <= min(mags) and max(mags) <= hi


def _fly_until_crash(s, action, frames=300):
    for _ in range(frames):
        if s.step(action):
            return s.crash_cause
    return None


def test_crash_causes():
    assert _fly_until_crash(empty_sim(2.0), C.DESCEND) == "ground"
    assert _fly_until_crash(empty_sim(8.0), C.ASCEND) == "ceiling"
    s = empty_sim(3.0)
    s.place_static(C.BUILDING, s.x + 2, s.x + 4, 0, 5)
    assert _fly_until_crash(s, C.HOLD) == "building"
    s = empty_sim(3.0)
    s.place_static(C.MOUNTAIN, s.x + 2, s.x + 10, 0, 5)
    assert _fly_until_crash(s, C.HOLD) == "mountain"
    s = empty_sim(7.0)
    s.place_static(C.STORM, s.x + 2, s.x + 8, 6.5, 10)
    assert _fly_until_crash(s, C.HOLD) == "storm"
    s = empty_sim(5.0)
    s.place_mover(C.BIRD, s.x + 3, 5.0, 0.5, 0.3, s=2.0)
    assert _fly_until_crash(s, C.HOLD) == "bird"
    s = empty_sim(5.0)
    s.place_mover(C.TRAFFIC, s.x + 3, 5.0, 1.0, 0.4, s=-1.0)
    assert _fly_until_crash(s, C.HOLD) == "traffic"


def test_shapes_are_respected():
    # over a building, under a storm dip, along a mountain slope: no crash
    s = empty_sim(6.0)
    s.place_static(C.BUILDING, s.x + 2, s.x + 4, 0, 5.7)
    s.place_static(C.STORM, s.x + 6, s.x + 12, 6.5, 10)
    s.place_static(C.MOUNTAIN, s.x + 14, s.x + 24, 0, 5.5)   # peak 5.5 < bottom 5.825
    assert _fly_until_crash(s, C.HOLD, 200) is None
    # mountain flank: the triangle, not its bounding box, collides
    s = empty_sim(1.5)
    s.place_static(C.MOUNTAIN, s.x + 1.0, s.x + 11.0, 0, 5.0)  # height 1.5 m about 2.6 m into it
    for _ in range(3):
        assert not s.step(C.HOLD)
    assert _fly_until_crash(s, C.HOLD) == "mountain"


def test_duck_shrinks_hitbox():
    def run(first):
        s = empty_sim(5.0)
        s.place_mover(C.BIRD, s.x + 0.6, 5.175 + 0.15 - 0.03, 0.5, 0.3)   # bird bottom 0.03 below top
        return _fly_until_crash(s, first if first is not None else C.HOLD, 20)
    assert run(C.HOLD) == "bird"
    s = empty_sim(5.0)
    s.place_mover(C.BIRD, s.x + 0.6, 5.175 + 0.15 - 0.03, 0.5, 0.3)
    s.step(C.DUCK)
    assert _fly_until_crash(s, C.HOLD, 20) is None


def test_step_after_crash_is_a_noop():
    s = empty_sim(0.4)
    assert _fly_until_crash(s, C.DESCEND) == "ground"
    h = s.state_hash()
    assert s.step(C.ASCEND) is True and s.state_hash() == h
