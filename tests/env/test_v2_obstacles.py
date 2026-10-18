"""SPEC v2.0 obstacles built from boxes and wires: trees, street lights, power lines, kites, cranes.
Their parts are the hitbox (collision), what the sensors see, and what the generator keeps passable."""
import numpy as np
import pytest

from drone_env import constants as C
from drone_env.generator import BOX, WIRE, parts_band
from drone_env.sensors import observe
from tests.env.helpers import empty_sim

V2 = (C.TREE, C.POLE, C.POWERLINE, C.KITE, C.CRANE)


def sample(kind, xs=10.0, seed=0):
    sim = empty_sim(5.0, "L1")
    return sim, sim.gen._sample_parts(kind, xs, np.random.default_rng(seed))


def put(sim, h):
    h.uid = 999
    sim.statics = [h]
    sim.pending = [h]
    return h


@pytest.mark.parametrize("kind", V2)
@pytest.mark.parametrize("seed", range(5))
def test_parts_bounds_and_band_hull(kind, seed):
    _, h = sample(kind, 10.0, seed)
    xs = [p[i] - (p[5] if p[0] == WIRE else 0) for p in h.parts for i in (1, 3)]
    assert min(xs) == pytest.approx(10.0) and h.x0 == pytest.approx(10.0)
    assert max(p[i] + (p[5] if p[0] == WIRE else 0) for p in h.parts for i in (1, 3)) == pytest.approx(h.x1)
    lo, hi = parts_band(h, h.x0 - 1, h.x1 + 1)
    assert max(lo, 0.0) == pytest.approx(h.y0, abs=1e-9) and hi == pytest.approx(h.y1, abs=1e-9)   # a kite string ends in the ground
    assert 0.0 <= h.y0 < h.y1 < C.CEILING
    assert h.x0 <= h.xc <= h.x1


def crash_at(h, x, y):
    """Crash cause (or None) with the drone centre at world (x, y) and the hazard placed."""
    sim = empty_sim(y, "L1")
    sim.distance_m = x - C.DRONE_X
    put(sim, h)
    sim._collide()
    return sim.crash_cause


def test_tree_canopy_and_trunk_block_the_top_is_clear():
    _, h = sample(C.TREE, 10.0, 1)
    trunk, mid = h.parts[0], h.parts[2]
    assert crash_at(h, 0.5 * (mid[1] + mid[3]), 0.5 * (mid[2] + mid[4])) == "tree"
    assert crash_at(h, 0.5 * (trunk[1] + trunk[3]), 0.5 * trunk[4]) == "tree"
    assert crash_at(h, 0.5 * (mid[1] + mid[3]), h.y1 + C.HALF_H + 0.05) is None


def test_powerline_under_and_over_are_clear_the_wires_are_not():
    _, h = sample(C.POWERLINE, 10.0, 2)
    xm = 0.5 * (h.x0 + h.x1)
    assert crash_at(h, xm, h.y0 - C.HALF_H - 0.05) is None
    assert crash_at(h, xm, h.y1 + C.HALF_H + 0.05) is None
    top_mid = parts_band(h, xm, xm)[1]
    assert crash_at(h, xm, top_mid) == "powerline"


def test_crane_under_the_jib_is_clear_but_not_at_the_hook():
    _, h = sample(C.CRANE, 10.0, 3)
    jib, cable, hook = h.parts
    x_away = jib[1] + 0.5 if abs(jib[1] + 0.5 - cable[1]) > 1.5 else jib[3] - 0.5
    y_mid = 0.5 * (hook[4] + jib[2])
    assert crash_at(h, x_away, y_mid) is None                          # under the jib, away from the hook
    assert crash_at(h, cable[1], y_mid) == "crane"                     # the hook cable
    assert crash_at(h, cable[1], hook[2] - C.HALF_H - 0.05) is None    # under the hook block
    assert crash_at(h, x_away, 0.5 * (jib[2] + jib[4])) == "crane"     # the jib itself


def test_kite_string_blocks_below_the_kite_and_above_is_clear():
    _, h = sample(C.KITE, 10.0, 4)
    string = h.parts[-1]
    xs_mid, ys_mid = 0.5 * (string[1] + string[3]), 0.5 * (string[2] + string[4])
    assert crash_at(h, xs_mid, ys_mid) == "kite"
    assert crash_at(h, h.xc, 0.5 * (h.parts[0][2] + h.parts[1][4])) == "kite"   # the kite itself
    assert crash_at(h, h.xc, h.y1 + C.HALF_H + 0.05) is None


def test_street_light_pole_blocks():
    _, h = sample(C.POLE, 10.0, 5)
    pole = h.parts[0]
    assert crash_at(h, 0.5 * (pole[1] + pole[3]), 0.5 * pole[4]) == "pole"
    assert crash_at(h, 0.5 * (pole[1] + pole[3]), h.y1 + C.HALF_H + 0.05) is None


@pytest.mark.parametrize("kind", V2)
def test_sensors_token_and_forward_ray(kind):
    sim, h = sample(kind, C.DRONE_X + C.HALF_W + 6.0, 6)
    put(sim, h)
    lo, hi = parts_band(h, h.x0, h.x1)
    o = observe(sim)
    m = o["hazard_mask"].astype(bool)
    assert m.sum() == 1 and o["hazard_type"][0] == kind
    gap, y0, y1 = o["hazards"][0, 0] * 16, o["hazards"][0, 1] * 10, o["hazards"][0, 2] * 10
    assert gap == pytest.approx(6.0, abs=1e-5) and y0 == pytest.approx(h.y0, abs=1e-5) and y1 == pytest.approx(h.y1, abs=1e-5)
    # straight-ahead ray at a height where the obstacle is solid at its first blocking part
    first = min(h.parts, key=lambda p: min(p[1], p[3]))
    if first[0] == BOX:
        sim.y = 0.5 * (first[2] + first[4])
        mid = C.N_RAYS // 2
        assert observe(sim)["rays"][mid] * 16 == pytest.approx(6.0 - C.HALF_W + C.HALF_W, abs=0.05) or \
            observe(sim)["rays"][mid] * 16 < 16.0


def test_v2_kinds_are_drawn(tmp_path):
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from drone_env.render import WorldPainter
    import pygame
    sim = empty_sim(5.0, "L1")
    x = C.DRONE_X + 3.0
    for i, kind in enumerate(V2):
        h = sim.gen._sample_parts(kind, x, np.random.default_rng(i))
        h.uid = 100 + i
        sim.statics.append(h)
        x = h.x1 + 1.0
    p = WorldPainter(scale=1.0)
    surf = pygame.Surface((p.w, p.h))
    p.draw(surf, sim)
    for h in sim.statics:
        assert p._sprite(h).get_height() == p.h
