"""SPEC v2.3: less predictable movers and the object tracker (drone_env/tracker.py)."""
import math
from dataclasses import replace

import numpy as np
import pytest

from drone_env import constants as C
from drone_env import tracker as T
from drone_env.generator import Mover
from drone_env.sim import Simulator
from teacher.planner import TeacherConfig, plan, tracked_envelopes

from .helpers import empty_sim


def generated_movers(kind, n=300, seed=0):
    gen = Simulator("L1", 0, generate=False).gen
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        out += gen._sample(kind, 0.0, rng)[1]
    return out


@pytest.mark.parametrize("kind", [C.BIRD, C.TRAFFIC])
def test_mover_paths_stay_in_their_band_and_vy_is_the_slope(kind):
    ts = np.linspace(0.0, 30.0, 901)
    divers = 0
    for m in generated_movers(kind):
        ys = np.array([m.y_at(t) for t in ts])
        assert ys.min() >= m.base - m.down - 1e-9 and ys.max() <= m.base + m.up + 1e-9
        assert ys.max() + m.h / 2 < C.CEILING and ys.min() - m.h / 2 > 0.0
        assert m.up + m.down <= T.SPAN[T.GROUP[kind]] + 1e-9
        for t in (1.3, 7.7, 21.1):
            num = (m.y_at(t + 1e-5) - m.y_at(t - 1e-5)) / 2e-5
            assert m.vy_at(t) == pytest.approx(num, abs=1e-3)
        divers += m.dive != 0.0
        assert m.fm > 0.0
    if kind == C.BIRD:
        assert 0.15 < divers / 300 < 0.45          # dive_prob 0.3
    else:
        assert divers == 0


def test_plane_paths_are_level():
    for m in generated_movers(C.PLANE, 20):
        assert m.still and m.y_at(3.0) == m.base and m.vy_at(3.0) == 0.0


def watched_bird(frames, gap=19.0, s=0.0, **path):
    sim = empty_sim(5.0, "L1")
    sim.place_mover(C.BIRD, sim.x + C.HALF_W + gap, 6.0, 0.5, 0.3, s=s, amp=1.0, period=1.5, **path)
    for _ in range(frames):
        sim.step(C.HOLD)
    return sim


def test_the_band_narrows_the_longer_a_bird_is_watched_and_holds_it():
    """A bird 19 m ahead (first seen at once), 6 m centre, +-1 m, 1.5 s swing: at first sight the predicted
    band is several metres tall; it narrows as the tracker sees the swing, the seen range grows to the
    swing, and while the drone passes, the bird stays inside the band."""
    heights, ages = [], []
    for f in (0, 6, 15, 30):
        sim = watched_bird(f)
        uid = sim.movers[0].uid
        age, s_lo, s_hi, p_lo, p_hi, _vy, g_lo, g_hi = T.predict(sim)[uid]
        assert p_lo <= g_lo <= g_hi <= p_hi                     # the best guess lies inside the band
        ages.append(age)
        heights.append(p_hi - p_lo)
    assert ages == pytest.approx([0.0, 0.2, 0.5, 1.0])
    assert heights[0] > 5.0 and heights[1] < heights[0] and heights[-1] < 2.0
    sim = watched_bird(0)
    misses = 0
    while sim.movers and sim.frame < 120:
        m = sim.movers[0]
        p = T.predict(sim).get(abs(m.uid))
        if p is None:
            break
        t = sim.frame * C.DT
        x0 = m.xr - m.s * (t - m.tr)
        ta, tb = T.window(sim, m, x0)
        if ta == 0.0:                               # passing it now: its body is inside the band
            y = m.y_at(t)
            misses += not (p[3] - 1e-6 <= y - m.h / 2 and y + m.h / 2 <= p[4] + 1e-6)
        sim.step(C.HOLD)
    assert misses == 0
    seen = T.predict(watched_bird(45))
    assert seen and all(v[2] - v[1] == pytest.approx(2.0 + 0.3, abs=0.02) for v in seen.values())


def test_the_tracker_knows_only_what_it_has_seen():
    """Two birds with the same sightings so far get the same prediction, whatever their hidden path
    settings: one will dive later, the other never does."""
    phase_d = -math.pi / 2                         # sin(omega_d t + phase_d) < 0 for the first 2 s
    a = watched_bird(20)
    b = watched_bird(20, dive=-0.8, omega_d=2 * math.pi / 8.0, phase_d=phase_d)
    ma, mb = a.movers[0], b.movers[0]
    assert [ma.y_at(t) for t in np.arange(0, 0.7, 0.05)] == pytest.approx([mb.y_at(t) for t in np.arange(0, 0.7, 0.05)])
    assert T.predict(a)[ma.uid] == pytest.approx(T.predict(b)[mb.uid])
    assert ma.down != mb.down                     # ... although the game knows b can go lower


def test_tracks_follow_the_sensors_and_clones():
    sim = watched_bird(10)
    uid = sim.movers[0].uid
    assert uid in sim.tracks and len(sim.tracks[uid].ts) == 11
    keep = sim.tracks
    c = sim.clone(calm=False)
    c.step(C.HOLD)
    assert sim.tracks is keep and len(sim.tracks[uid].ts) == 11 and len(c.tracks[uid].ts) == 12
    pc = sim.clone(calm=True)                      # planner copies do not track
    pc.step(C.HOLD)
    assert pc.tracking is False and pc.tracks is keep
    from drone_env import events
    fog = watched_bird(0, gap=12.0)                # in fog (8 m) a bird 12 m ahead is not seen: no track
    events.start(fog, events.FOG, 5.0)
    fog.step(C.HOLD)
    assert fog.tracks == {}
    gone = watched_bird(0, gap=1.0)                # passed: the track is dropped
    for _ in range(30):
        gone.step(C.DESCEND)
    assert gone.tracks == {}


def test_teacher_plans_seen_movers_as_the_tracked_band():
    sim = watched_bird(15)
    m = sim.movers[0]
    p = T.predict(sim)[m.uid]
    c = sim.clone(calm=True)
    tracked_envelopes(c, sim)
    cm = c.movers[0]
    assert cm.amp == 0.0 and cm.dive == 0.0 and cm.fm == 0.0
    assert cm.base - cm.h / 2 == pytest.approx(p[3]) and cm.base + cm.h / 2 == pytest.approx(p[4])
    assert sim.movers[0] is m and m.amp == 1.0          # the sim itself is untouched
    cfg = TeacherConfig(sensor_limited=True)
    a = plan(sim, replace(cfg, tracked_movers=True))
    b = plan(sim, replace(cfg, tracked_movers=True))
    assert np.array_equal(a.scores, b.scores)            # pure function of the sim
    g = sim.clone(calm=True)
    tracked_envelopes(g, sim, best_guess=True)
    gm = g.movers[0]
    assert gm.base - gm.h / 2 == pytest.approx(p[6]) and gm.base + gm.h / 2 == pytest.approx(p[7])


def test_boxed_in_by_the_bands_the_teacher_plans_with_its_best_guess():
    """A level drone (no burst: cooldown) 0.07 s before passing a hovering traffic drone whose band's
    margin reaches over the drone's top: no move gets clear of the band in time, but the best guess
    (0.1 m above the drone) is clear; with the fallback the teacher plans against that instead."""
    sim = empty_sim(5.0, "L1")
    sim.place_mover(C.TRAFFIC, sim.x + C.HALF_W + 9.0, 5.475, 1.0, 0.4, s=0.0)
    for _ in range(25):
        sim.step(C.HOLD)
    sim.cooldown = 30
    p = T.predict(sim)[sim.movers[0].uid]
    top = sim.y + C.HALF_H
    assert p[3] < top < p[6]                             # inside the band's margin, under the best guess
    cfg = TeacherConfig(sensor_limited=True, tracked_movers=True)
    boxed = plan(sim, cfg)
    fb = plan(sim, replace(cfg, tracked_fallback=True))
    assert not boxed.safe.any() and fb.safe[C.HOLD]


def test_v23_hazard_rows_carry_the_tracker():
    sim = watched_bird(15)
    o = sim.observe()
    p = T.predict(sim)[sim.movers[0].uid]
    row = o["hazards"][0]
    assert o["hazard_type"][0] == C.BIRD
    assert row[8] == pytest.approx(min(p[0] / 2.0, 1.0))
    assert row[9:13] * 10 == pytest.approx(p[1:5], abs=1e-4)
    assert row[5] * 8 == pytest.approx(p[5], abs=1e-4)                  # vy from the last two sightings
    m = sim.movers[0]
    y = m.y_at(sim.frame * C.DT)
    assert row[1] * 10 == pytest.approx(y - 0.15, abs=1e-4) and row[2] * 10 == pytest.approx(y + 0.15, abs=1e-4)


def test_margins_file_matches_the_bins():
    mg = T.margins()
    assert mg.shape == (3, len(T.AGE_EDGES) + 1, len(T.AHEAD_EDGES) + 1, 2)
    assert (mg[:2] >= 0.02 - 1e-9).all() and (mg[2] == 0).all()
    # younger tracks never get smaller margins than older ones; further ahead never smaller than nearer
    assert (np.diff(mg[:2], axis=1) <= 1e-9).all() and (np.diff(mg[:2], axis=2) >= -1e-9).all()
    # between cells the margin is interpolated: no step at a cell edge
    a = T.margin(T.G_BIRD, 0.299, 0.7)
    b = T.margin(T.G_BIRD, 0.301, 0.7)
    assert np.abs(a - b).max() < 0.02
    assert T.margin(T.G_BIRD, T.AGE_MIDS[4], T.AHEAD_MIDS[3]) == pytest.approx(mg[0, 4, 3])
