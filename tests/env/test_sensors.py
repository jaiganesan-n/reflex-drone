import math

import gymnasium as gym
import numpy as np
import pytest

import drone_env  # noqa: F401
from drone_env import constants as C
from drone_env import events
from drone_env.sim import Simulator
from drone_env.wrappers import FlatHistory, FlattenObs, FrameStack, flatten_obs, stack_frames

from .helpers import empty_sim, scripted


def rollout_obs(level, seed, frames=600, every=7):
    s = Simulator(level, seed)
    out = []
    for f in range(frames):
        if f % every == 0:
            out.append((s.observe(), s.fog_active))
        if s.step(scripted(s, f)):
            break
    return out


@pytest.mark.parametrize("level", ["L1", "L3", "L5", "stress"])
def test_shapes_dtypes_ranges(level):
    for seed in (100_000, 100_001, 100_002):
        for o, fog in rollout_obs(level, seed):
            assert o["self"].shape == (11,) and o["self"].dtype == np.float32
            assert o["rays"].shape == (13,) and o["rays"].dtype == np.float32
            assert o["hazards"].shape == (12, 13) and o["hazards"].dtype == np.float32     # SPEC v2.3 rows
            assert o["hazard_type"].shape == (12,) and o["hazard_type"].dtype == np.int64
            assert o["hazard_mask"].shape == (12,) and o["hazard_mask"].dtype == np.int8
            assert o["action_mask"].shape == (5,) and o["action_mask"].dtype == np.int8
            assert ((o["rays"] >= 0) & (o["rays"] <= (0.5 if fog else 1.25))).all()   # 20 m / 16
            m = o["hazard_mask"].astype(bool)
            hz = o["hazards"]
            assert ((hz[m, 0] >= 0) & (hz[m, 0] <= 1.25)).all()            # dx in [0, 20 m / 16]
            assert (o["hazard_type"][m] >= 1).all() and (o["hazard_type"][m] <= C.N_HAZARD_TYPES - 1).all()
            assert (hz[~m] == 0).all() and (o["hazard_type"][~m] == 0).all()
            assert (hz[m, 3] <= 1.5).all() and (hz[m, 3] > 0).all()
            assert (hz[m, 6] >= 0).all() and (hz[m, 6] <= 1.6).all()
            assert set(np.unique(hz[m, 7])) <= {0.0, 1.0}
            # tracker values: watched in [0, 1] (statics 1); the body now lies in the seen range;
            # still hazards are fully known (seen = predicted = their bounds)
            assert ((hz[m, 8] >= 0) & (hz[m, 8] <= 1)).all()
            assert (hz[m, 9] <= hz[m, 1] + 1e-3).all() and (hz[m, 2] <= hz[m, 10] + 1e-3).all()
            assert (hz[m, 11] <= hz[m, 12] + 1e-6).all()
            st = m & (hz[:, 7] == 0)
            assert (hz[st, 8] == 1).all() and np.allclose(hz[st, 9:11], hz[st, 1:3]) and np.allclose(hz[st, 11:13], hz[st, 1:3])
            assert o["self"][6:].sum() == 1.0 and 0.0 <= o["self"][0] <= 1.0
            if fog:
                assert (hz[m, 0] <= 0.5 + 1e-6).all()


def test_self_vector():
    s = empty_sim(5.0)
    s.step(C.JUMP)
    o = s.observe()
    assert o["self"][0] == pytest.approx(s.y / 10)
    assert o["self"][1] == pytest.approx(s.vy / 8)
    assert o["self"][2] == 0.0
    assert o["self"][3] == pytest.approx(29 / 30)
    assert o["self"][4] == pytest.approx(0.25)                # (10 - 8) / 8: L1 flies at 10 m/s (v2.3)
    assert o["self"][6 + C.JUMP] == 1.0
    s = empty_sim(5.0)
    s.step(C.DUCK)
    assert s.observe()["self"][2] == 1.0


def test_ray_geometry():
    s = empty_sim(4.0)
    o = s.observe()
    # no hazards: the -30 deg ray hits the ground at 8 m, the +30 deg ray the ceiling at 12 m
    i_m30 = C.RAY_ANGLES_DEG.index(-30.0)
    i_p30 = C.RAY_ANGLES_DEG.index(30.0)
    i_0 = C.RAY_ANGLES_DEG.index(0.0)
    assert o["rays"][i_m30] * 16 == pytest.approx(8.0, abs=1e-4)
    assert o["rays"][i_p30] * 16 == pytest.approx(12.0, abs=1e-4)
    assert o["rays"][i_0] == pytest.approx(20 / 16)          # SPEC v2.3: 20 m range
    s.place_static(C.BUILDING, s.x + 5.0, s.x + 7.0, 0.0, 6.0)
    o = s.observe()
    assert o["rays"][i_0] * 16 == pytest.approx(5.0, abs=1e-4)          # centre to the wall
    s = empty_sim(4.0)
    s.place_static(C.STORM, s.x + 2.0, s.x + 10.0, 6.5, 10.0)
    o = s.observe()
    assert o["rays"][i_p30] * 16 < 12.0 - 1e-3                          # the dip is hit first
    s = empty_sim(1.0)
    s.place_static(C.MOUNTAIN, s.x + 3.0, s.x + 11.0, 0.0, 6.0)
    o = s.observe()
    assert o["rays"][i_0] * 16 == pytest.approx(3.0 + 1.0 * 4.0 / 6.0, abs=1e-3)


def test_hazard_token_features():
    s = empty_sim(4.0)
    b = s.place_static(C.BUILDING, s.x + 5.45, s.x + 7.45, 0.0, 6.0)
    o = s.observe()
    m = o["hazard_mask"].astype(bool)
    assert m.sum() == 1 and o["hazard_type"][0] == C.BUILDING
    dx, yb, yt, w, vx, vy, dist, moving, watched, s_lo, s_hi, p_lo, p_hi = o["hazards"][0]
    assert dx * 16 == pytest.approx(5.0, abs=1e-5)          # nose (x + 0.45) to left edge
    assert (yb, yt) == pytest.approx((0.0, 0.6)) and w == pytest.approx(2 / 8)
    assert vx == pytest.approx(-10 / 16) and vy == 0 and moving == 0    # L1 scroll speed 10 m/s (v2.3)
    assert watched == 1.0 and (s_lo, s_hi, p_lo, p_hi) == pytest.approx((0.0, 0.6, 0.0, 0.6))
    assert dist * 16 == pytest.approx(math.hypot(5.45, 0.0), abs=1e-4)
    assert s.last_obs_uids == (b.uid,)
    s = empty_sim(5.0)
    s.place_mover(C.TRAFFIC, s.x + 3, 5.0, 1.0, 0.4, s=2.0)
    hz = s.observe()["hazards"][0]
    assert hz[4] == pytest.approx(-(10 + 2) / 16) and hz[7] == 1.0
    # overlapping in x -> dx = 0
    s = empty_sim(8.0)
    s.place_static(C.BUILDING, s.x - 1.0, s.x + 1.0, 0.0, 5.0)
    assert s.observe()["hazards"][0, 0] == 0.0


def test_sensor_range_clear_and_fog():
    for gap, rng, fog, visible in [(19.0, 20, False, True), (20.5, 20, False, False),
                                   (7.5, 8, True, True), (8.5, 8, True, False)]:
        s = empty_sim(4.0)
        if fog:
            events.start(s, events.FOG, 5.0)
        nose = s.x + C.HALF_W
        s.place_static(C.BUILDING, nose + gap, nose + gap + 2, 0.0, 3.0)
        o = s.observe()
        assert bool(o["hazard_mask"].any()) == visible, (gap, fog)
    # behind the tail -> not visible
    s = empty_sim(4.0)
    s.place_static(C.BUILDING, s.x - 5.0, s.x - C.HALF_W - 0.01, 0.0, 3.0)
    assert not s.observe()["hazard_mask"].any()


def test_fog_caps_rays_and_adds_noise():
    s = empty_sim(4.0)
    clear = s.observe()["rays"]
    events.start(s, events.FOG, 5.0)
    a = s.observe()["rays"]
    b = s.observe()["rays"]
    assert (a <= 0.5).all() and (b <= 0.5).all()
    assert not np.array_equal(a, b)                       # sensor noise
    exp = np.minimum(clear, 0.5)
    assert np.abs(a - exp).max() < 0.15


def test_max_12_nearest_hazards():
    s = empty_sim(5.0)
    for i in range(20):
        s.place_mover(C.BIRD, s.x + 1 + 0.7 * i, 2.0 + (i % 3) * 3, 0.5, 0.3)
    o = s.observe()
    assert o["hazard_mask"].sum() == 12
    d = o["hazards"][:, 6]
    all_d = sorted(math.hypot(max(s.x + 1 + 0.7 * i - s.x, 0), max(abs(2.0 + (i % 3) * 3 - 5.0) - 0.15, 0)) / 16
                   for i in range(20))
    assert np.allclose(np.sort(d), all_d[:12], atol=1e-5)


# ------------------------------------------------------------------ flatten_obs --------
def batch_obs(n=64):
    obs = []
    for level, seed in [("L5", 100_000), ("stress", 300_000), ("L3", 100_004)]:
        obs += [o for o, _ in rollout_obs(level, seed, 900, 5)]
    obs = obs[:n]
    keys = ("self", "rays", "hazards", "hazard_type", "hazard_mask")
    return {k: np.stack([o[k] for o in obs]) for k in keys}


def test_flatten_shapes_and_padding():
    b = batch_obs()
    out = flatten_obs(b["self"], b["rays"], b["hazards"], b["hazard_type"], b["hazard_mask"])
    assert out.shape == (len(b["self"]), 40) and out.dtype == np.float32
    s = empty_sim(5.0)
    o = s.observe()
    f = flatten_obs(o["self"], o["rays"], o["hazards"], o["hazard_type"], o["hazard_mask"])
    assert f.shape == (1, 40)
    assert (f[0, 19:] == np.array([1, 0, 0, 0, 0, 0, 0] * 3)).all()
    assert (f[0, :6] == o["self"][:6]).all() and (f[0, 6:19] == o["rays"]).all()


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_flatten_batched_equals_per_frame(dtype):
    b = batch_obs()
    args = [b["self"].astype(dtype), b["rays"].astype(dtype), b["hazards"].astype(dtype),
            b["hazard_type"], b["hazard_mask"]]
    batched = flatten_obs(*args)
    for i in range(len(batched)):
        single = flatten_obs(*[a[i:i + 1] for a in args])
        assert np.array_equal(single[0], batched[i])


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_flatten_ignores_hazard_order(dtype):
    b = batch_obs()
    rng = np.random.default_rng(0)
    args = [b["self"].astype(dtype), b["rays"].astype(dtype), b["hazards"].astype(dtype),
            b["hazard_type"], b["hazard_mask"]]
    ref = flatten_obs(*args)
    for _ in range(5):
        perm = np.stack([rng.permutation(12) for _ in range(len(ref))])
        shuf = [args[0], args[1], np.take_along_axis(args[2], perm[..., None], 1),
                np.take_along_axis(args[3], perm, 1), np.take_along_axis(args[4], perm, 1)]
        assert np.array_equal(flatten_obs(*shuf), ref)
    # ties in dx are broken by the other features, not by row order
    hz = np.zeros((1, 12, 8), dtype)
    ht = np.zeros((1, 12), np.int64)
    hm = np.zeros((1, 12), np.int8)
    for i in range(4):
        hz[0, i] = [0.25, 0.1 * i, 0.2 + 0.1 * i, 0.2, -0.4, 0, 0.3, 0]
        ht[0, i], hm[0, i] = 1, 1
    s, r = np.zeros((1, 11), dtype), np.zeros((1, 13), dtype)
    a = flatten_obs(s, r, hz, ht, hm)
    p = np.array([[3, 1, 0, 2] + list(range(4, 12))])
    bb = flatten_obs(s, r, np.take_along_axis(hz, p[..., None], 1), np.take_along_axis(ht, p, 1),
                     np.take_along_axis(hm, p, 1))
    assert np.array_equal(a, bb)


def test_flatten_feature_order():
    hz = np.zeros((1, 12, 8), np.float32)
    ht = np.zeros((1, 12), np.int64)
    hm = np.zeros((1, 12), np.int8)
    hz[0, 5] = [0.5, 0.31, 0.42, 0.25, -0.4, 0.11, 0.6, 1.0]    # bird, further
    ht[0, 5], hm[0, 5] = 3, 1
    hz[0, 2] = [0.1, 0.0, 0.6, 0.3, -0.4, 0.0, 0.2, 0.0]        # building, nearest
    ht[0, 2], hm[0, 2] = 1, 1
    out = flatten_obs(np.zeros((1, 11)), np.zeros((1, 13)), hz, ht, hm)[0]
    assert out[19:26] == pytest.approx([0.1, 0.6, 0.0, 0.3, 0.0, 0.0, 0.0])
    assert out[26:33] == pytest.approx([0.5, 0.42, 0.31, 0.25, 0.11, 1.0, 1.0])
    assert out[33:40] == pytest.approx([1, 0, 0, 0, 0, 0, 0])


def test_wrappers_framestack():
    env = FrameStack(FlattenObs(gym.make("DroneDash-v0", level="L3")), 2)
    assert env.observation_space.shape == (80,)
    o0, _ = env.reset(seed=4)
    assert o0.shape == (80,) and np.array_equal(o0[:40], o0[40:])
    o1, *_ = env.step(C.ASCEND)
    assert np.array_equal(o1[:40], o0[40:])
    o2, *_ = env.step(C.ASCEND)
    assert np.array_equal(o2[:40], o1[40:])
    # dict env is flattened automatically; FlatHistory follows the same rule
    raw = gym.make("DroneDash-v0", level="L3")
    st = FrameStack(raw, 2)
    p0, _ = st.reset(seed=4)
    assert np.array_equal(p0, o0)
    h = FlatHistory()
    env2 = gym.make("DroneDash-v0", level="L3")
    d0, _ = env2.reset(seed=4)
    assert np.array_equal(h.push(d0), o0)
    d1, *_ = env2.step(C.ASCEND)
    assert np.array_equal(h.push(d1), o1)
    assert np.array_equal(stack_frames(o0[40:], o1[40:]), o1)


@pytest.mark.parametrize("spec", [(2, 1), (2, 2), (2, 3)])
def test_sensor_spec_rows_for_a_bobbing_mover(spec):
    """Each pilot gets the rows it was made for. v2.2: a bobbing bird shows as the whole band its path
    covers (base -/+ (amp + h/2), from the game's settings); before v2.2: its body now. v2.3: its body
    now plus 5 tracker values; at first sight it has been watched 0 s, seen only where it is, and its
    predicted band is wide. Rays, gap, width, closing speed, distance and the moving flag are the same."""
    from drone_env.env import DroneDashEnv

    def build(sp):
        sim = empty_sim(5.0, "L1")
        sim.sensor_spec = sp
        sim.place_mover(C.BIRD, sim.x + C.HALF_W + 6.0, 6.0, 0.5, 0.3, s=2.0, amp=1.0, period=1.5)
        sim.place_mover(C.TRAFFIC, sim.x + C.HALF_W + 9.0, 3.0, 1.0, 0.4, s=0.0)
        return sim, sim.observe()
    sim, o = build(spec)
    assert o["hazards"].shape == (12, 13 if spec >= (2, 3) else 8)
    my = sim.movers[0].y_at(sim.frame * C.DT)
    rows = {int(t): o["hazards"][i] for i, t in enumerate(o["hazard_type"]) if o["hazard_mask"][i]}
    want = (6.0 - 1.0 - 0.15, 6.0 + 1.0 + 0.15) if spec == (2, 2) else (my - 0.15, my + 0.15)
    assert rows[C.BIRD][1] * 10 == pytest.approx(want[0], abs=1e-5)
    assert rows[C.BIRD][2] * 10 == pytest.approx(want[1], abs=1e-5)
    assert rows[C.TRAFFIC][1] * 10 == pytest.approx(2.8, abs=1e-5) and rows[C.TRAFFIC][2] * 10 == pytest.approx(3.2, abs=1e-5)
    if spec >= (2, 3):
        b = rows[C.BIRD]
        assert b[8] == 0.0 and b[9:11] * 10 == pytest.approx(want, abs=1e-5)
        assert b[11] * 10 < want[0] - 1.0 and b[12] * 10 > want[1] + 1.0
    _sim2, o2 = build((2, 2) if spec != (2, 2) else (2, 1))
    assert np.array_equal(np.minimum(o["rays"], 1.0), np.minimum(o2["rays"], 1.0))     # 16 m vs 20 m range
    for col in (0, 3, 4, 6, 7):
        assert np.array_equal(o["hazards"][:, col], o2["hazards"][:, col])
    env = DroneDashEnv("L1")
    env.sensor_spec = spec
    env.reset(seed=1)
    assert env.sim.sensor_spec == spec and env.sim.clone().sensor_spec == spec
    assert env.observation_space["hazards"].shape == (12, 13 if spec >= (2, 3) else 8)
    assert env.observation_space.contains(env.sim.observe())


def test_older_specs_see_16_m():
    for spec, gap, visible in [((2, 2), 15.0, True), ((2, 2), 17.0, False), ((2, 3), 17.0, True)]:
        s = empty_sim(4.0)
        s.sensor_spec = spec
        nose = s.x + C.HALF_W
        s.place_static(C.BUILDING, nose + gap, nose + gap + 2, 0.0, 3.0)
        o = s.observe()
        assert bool(o["hazard_mask"].any()) == visible, (spec, gap)
        assert o["rays"].max() <= C.sensor_range(spec) / 16 + 1e-6
