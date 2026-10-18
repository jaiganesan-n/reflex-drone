"""Simulator.path_time: seconds until a hazard blocks the drone's current height band (planner use)."""
import numpy as np
import pytest

from drone_env import constants as C
from tests.env.helpers import empty_sim


def nose(sim):
    return sim.distance_m + C.DRONE_X + C.HALF_W


def test_open_sky_is_clear_for_the_whole_cap():
    assert empty_sim(5.0, "L1").path_time(1.2) == 1.2


@pytest.mark.parametrize("y, expect", [(3.0, 0.8), (6.0, None)])
def test_building_ahead_blocks_below_its_roof(y, expect):
    """A building 0-5 m tall 8 m ahead at L1 (10 m/s, v2.3): at 3 m it is reached in 0.8 s; at 6 m the
    band (hitbox + 0.3 m) is above the roof and the path stays clear."""
    sim = empty_sim(y, "L1")
    sim.place_static(C.BUILDING, nose(sim) + 8.0, nose(sim) + 10.0, 0.0, 5.0)
    t = sim.path_time(2.0)
    assert t == pytest.approx(expect) if expect is not None else t == 2.0


def test_building_beyond_the_cap_is_not_counted():
    sim = empty_sim(3.0, "L1")
    sim.place_static(C.BUILDING, nose(sim) + 20.0, nose(sim) + 22.0, 0.0, 5.0)
    assert sim.path_time(1.2) == 1.2


def test_mover_closes_at_scroll_plus_its_own_speed():
    sim = empty_sim(5.0, "L1")
    sim.place_mover(C.BIRD, nose(sim) + 10.0, 5.0, 0.5, 0.3, s=2.0)
    assert sim.path_time(3.0) == pytest.approx(10.0 / (10.0 + 2.0))
    high = empty_sim(5.0, "L1")
    high.place_mover(C.BIRD, nose(high) + 10.0, 8.0, 0.5, 0.3, s=2.0)
    assert high.path_time(3.0) == 3.0


def test_power_line_blocks_only_at_wire_height():
    """Under the lowest point of the sag and over the poles' tops the path is clear; at wire height it is not."""
    sim = empty_sim(5.0, "L1")
    h = sim.gen._sample_parts(C.POWERLINE, nose(sim) + 4.0, np.random.default_rng(2))
    h.uid = 999
    sim.statics, sim.pending = [h], [h]
    m = C.HALF_H + 0.3 + 0.05
    for y, clear in ((h.y0 - m, True), (h.y1 + m, True), (0.5 * (h.y0 + h.y1), False)):
        sim.y = y
        t = sim.path_time(3.0)
        assert (t == 3.0) if clear else (t < 3.0)
