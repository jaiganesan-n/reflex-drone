import numpy as np
import pytest

from drone_env import constants as C
from drone_env.generator import Mover, Static, verify_layout
from drone_env.sim import Simulator

from .helpers import scripted


def layout(level, seed, upto):
    s = Simulator(level, seed, record=True)
    s.gen.generate_until(upto, s.statics, s.dormant)
    return s


def test_500_random_L5_layouts_pass_the_passability_check():
    rng = np.random.default_rng(2026)
    seeds = rng.choice(100_000, size=500, replace=False)
    worst = np.inf
    for seed in seeds:
        s = layout("L5", int(seed), 130.0)
        ok, fail_x, min_corr = verify_layout(s.gen.log, s.schedule, 120.0, dx=0.25, dy=0.025)
        assert ok, f"seed {seed}: impassable at x = {fail_x}"
        assert min_corr >= C.CORRIDOR_MIN - 0.05, f"seed {seed}: corridor {min_corr:.2f} m"
        worst = min(worst, min_corr)


@pytest.mark.parametrize("level", ["L1", "L3", "L5", "stress", "endless"])
def test_min_spacing_between_obstacle_starts(level):
    for seed in range(100_000, 100_010):
        s = layout(level, seed, 900.0)
        starts = s.gen.starts
        for (x0, v0), (x1, v1) in zip(starts, starts[1:]):
            assert x1 - x0 >= v0 * C.MIN_GAP_S - 1e-9


def test_min_spacing_past_l5_never_below_the_l5plus_floor():
    s = Simulator("endless", 100_003, record=True)
    s.gen.generate_until(C.DRONE_X + 13_000.0, s.statics, s.dormant)
    late = [(x, v) for x, v in s.gen.starts if x > C.DRONE_X + C.ENDLESS_RAMP_M]
    assert len(late) > 200
    for (x0, v0), (x1, _v1) in zip(late, late[1:]):
        assert x1 - x0 >= v0 * C.L5P_MIN_GAP_S - 1e-9


def test_world_generated_at_least_40_m_ahead():
    s = Simulator("L5", 100_007)
    for f in range(1500):
        assert s.gen.next_start >= s.x + C.LOOKAHEAD
        s.step(scripted(s, f))
        if s.crashed:
            break


def kinds(s, x_lo=-1e9, x_hi=1e9):
    out = set()
    for h in s.gen.log:
        x = h.x0 if isinstance(h, Static) else h.xe
        if x_lo <= x < x_hi:
            out.add(h.kind)
    return out


ALL_KINDS = {C.BUILDING, C.TREE, C.POLE, C.KITE, C.STORM, C.BIRD, C.TRAFFIC, C.PLANE, C.DOWNDRAFT}   # SPEC v2.1


def stats(level, seeds=range(5), upto=2500.0):
    events = hazards = combos = 0
    km = 0.0
    for seed in seeds:
        s = layout(level, seed, upto)
        events += s.gen.placed
        hazards += len(s.gen.log)
        combos += s.gen.combos
        km += (upto - C.DRONE_X) / 1000
    return events / km, hazards / km, combos / events


def test_every_level_has_every_obstacle_type():
    for level in ("L1", "L2", "L3", "L4", "L5", "stress"):
        seen = set().union(*(kinds(layout(level, seed, 2500)) for seed in (1, 2, 3)))
        assert seen == ALL_KINDS, level      # v2.1: no mountains, power lines or cranes


def test_density_and_combos_grow_with_the_level():
    rows = {lvl: stats(lvl) for lvl in ("L1", "L2", "L3", "L4", "L5")}
    assert rows["L1"][2] == 0.0                                   # no combos at L1
    per_km = [rows[l][1] for l in ("L1", "L2", "L3", "L4", "L5")]
    combo = [rows[l][2] for l in ("L1", "L2", "L3", "L4", "L5")]
    assert per_km == sorted(per_km) and per_km[-1] > 1.3 * per_km[0]
    assert combo == sorted(combo) and combo[-1] > 0.35


def test_endless_levels_start_every_1750_m_and_the_speed_ramps_over_l1():
    """SPEC v2.3: L1 10 m/s, L2-L5 12 m/s; levels start at equal distances, the speed moves from
    each level's speed to the next one's in between (10 -> 12 m/s over L1, then 12 m/s)."""
    sc = Simulator("endless", 0, generate=False).schedule
    assert [sc.distance_for_level(i) for i in range(5)] == pytest.approx([0.0, 1750.0, 3500.0, 5250.0, 7000.0])
    assert [sc.spec_at(d).name for d in (0.0, 1749.0, 1750.0, 3600.0, 5300.0, 7000.0, 12_000.0)] == \
        ["L1", "L1", "L2", "L3", "L4", "L5", "L5"]
    assert [sc.speed_at(d) for d in (0.0, 875.0, 1750.0, 4000.0, 9000.0)] == pytest.approx([10.0, 11.0, 12.0, 12.0, 12.0])


def test_endless_follows_the_speed():
    e = layout("endless", 2, 9000)
    e2 = layout("endless", 3, 9000)
    off = C.DRONE_X
    for lvl, (a, b) in {"L1": (0, 1750), "L3": (3500, 5250), "L5": (7000, 9000)}.items():
        assert kinds(e, a + off, b + off) | kinds(e2, a + off, b + off) == ALL_KINDS, lvl
    # spacing follows the speed: obstacle starts are further apart (in metres) later on
    st = e.gen.starts
    early = np.mean([x1 - x0 for (x0, _), (x1, _) in zip(st, st[1:]) if x1 < 1000])
    v_late = [v for x, v in st if x > 7500]
    assert early > 10.0 * 1.9 and min(v_late) == pytest.approx(12.0) and max(v_late) == pytest.approx(12.0)


def test_long_endless_run_keeps_generating_past_10km():
    s = Simulator("endless", 5)
    s.distance_m = 10_500.0                       # jump ahead: the generator catches up
    s._refill()
    s._update_local()
    assert s.gen.next_start >= s.x + C.LOOKAHEAD and s.gen.fallbacks == 0
    assert s.level_now == "L5" and s.speed == pytest.approx(12.0)


def per_km(starts, a, b):
    return sum(1 for x, _ in starts if a <= x - C.DRONE_X < b) / ((b - a) / 1000.0)


def test_l5plus_constant_speed_and_denser_obstacles():
    """SPEC v1.7: past 7 km the speed stays at L5's (12 m/s, v2.3) while obstacles get denser, and every
    obstacle still passes both solvability checks (independent raster check too)."""
    dens_early, dens_late = [], []
    for seed in (100_001, 100_002):
        s = Simulator("endless", seed, record=True)
        s.gen.generate_until(C.DRONE_X + 14_000.0, s.statics, s.dormant)
        assert s.schedule.speed_at(7_000.0) == pytest.approx(12.0)
        assert s.schedule.speed_at(13_500.0) == pytest.approx(12.0)
        lo, hi, combo = s.schedule.density_at(13_000.0, s.schedule.specs[-1])
        assert lo < 0.9 and combo > 0.7
        dens_early.append(per_km(s.gen.starts, 7_000.0, 8_000.0))
        dens_late.append(per_km(s.gen.starts, 12_000.0, 14_000.0))
        assert s.gen.fallbacks <= 5
        ok, fail_x, min_corr = verify_layout(s.gen.log, s.schedule, 13_900.0, dx=0.5, dy=0.05)
        assert ok, f"seed {seed}: impassable at x = {fail_x}"
    assert np.mean(dens_late) > 1.3 * np.mean(dens_early)


def test_hazard_parameters_in_range():
    s = layout("L5", 3, 3000)
    hz = C.HZ
    for h in s.gen.log:
        if isinstance(h, Static):
            w = h.x1 - h.x0
            if h.kind == C.BUILDING:
                assert hz["building"]["width"][0] <= w <= hz["building"]["width"][1]
                assert hz["building"]["height"][0] <= h.y1 <= hz["building"]["height"][1]
            elif h.kind == C.MOUNTAIN:
                assert 6.0 <= w <= 12.0 and 3.0 <= h.y1 <= 6.5
            elif h.kind == C.STORM:
                assert 4.0 <= w <= 8.0 and 6.5 <= h.y0 <= 8.5
            elif h.kind == C.DOWNDRAFT:
                assert 3.0 <= w <= 6.0
        else:
            assert isinstance(h, Mover)
            if h.kind == C.BIRD:
                assert 3.0 <= h.base <= 9.0 and h.amp <= 1.5 and 1.0 <= h.s <= 3.0
                assert h.base + h.amp + h.h / 2 < C.CEILING
            elif h.kind == C.PLANE:
                assert 8.0 <= h.base <= 9.5 and 4.0 <= h.s <= 8.0 and h.amp == 0.0   # v1.9: planes fly high
                assert (h.sub == 0 and 4.5 <= h.w <= 5.5 and h.h == 1.0) or (h.sub == 1 and 3.0 <= h.w <= 3.6 and h.h == 0.8)
                assert h.base + h.h / 2 < C.CEILING
            else:
                assert 3.0 <= h.base <= 9.0 and -2.0 <= h.s <= 4.0


def test_generator_is_deterministic():
    a, b = layout("L5", 9, 600), layout("L5", 9, 600)
    assert [h.key() for h in a.gen.log] == [h.key() for h in b.gen.log]
    c = layout("L5", 10, 600)
    assert [h.key() for h in a.gen.log] != [h.key() for h in c.gen.log]


def test_traffic_drifts_slowly_and_stays_under_the_ceiling():
    """SPEC v2.1: drones and balloons drift up and down (0.3-0.6 m, one cycle per 2-4 s) instead of flying
    dead level; planes still fly level; nothing leaves the sky."""
    import math
    seen = 0
    for level, seed in (("L1", 1), ("L3", 2), ("L5", 3)):
        s = layout(level, seed, 2500)
        for m in s.dormant + s.movers:
            if m.kind == C.TRAFFIC:
                seen += 1
                assert 0.0 <= m.amp <= 0.6 + 1e-9 and 2.0 <= 2 * math.pi / m.omega <= 4.0 + 1e-9
                assert m.base + m.amp + m.h / 2 <= C.CEILING - 0.1 + 1e-9
            elif m.kind == C.PLANE:
                assert m.amp == 0.0
    assert seen > 10
