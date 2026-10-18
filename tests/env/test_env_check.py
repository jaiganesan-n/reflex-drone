import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import drone_env  # noqa: F401
from drone_env import constants as C

INFO_KEYS = {"frame", "distance_m", "speed", "level", "crash_cause", "gust_active", "surge_active",
             "fog_active", "hazards_visible", "masked_action_used"}


@pytest.mark.parametrize("level", C.LEVELS)
def test_check_env_passes(level):
    env = gym.make("DroneDash-v0", level=level)
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # any checker warning fails the test
        check_env(env.unwrapped)
    env.close()


def test_make_kwargs_and_spaces():
    env = gym.make("DroneDash-v0", level="L3", render_mode=None, max_frames=10_000)
    obs, info = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    assert env.action_space.n == 5
    assert INFO_KEYS <= set(info)
    assert env.unwrapped.sim.level == "L3" and env.unwrapped.sim.seed == 123


def test_truncation_and_unlimited():
    env = gym.make("DroneDash-v0", level="L1", max_frames=50)
    env.reset(seed=1)
    for i in range(50):
        _, _, term, trunc, info = env.step(C.HOLD)
    assert not term and trunc and info["frame"] == 50
    env = gym.make("DroneDash-v0", level="L1", max_frames=None)
    env.reset(seed=1)
    for i in range(60):
        _, _, term, trunc, _ = env.step(C.HOLD)
        assert not trunc


def test_rewards():
    env = gym.make("DroneDash-v0", level="L1", max_frames=None).unwrapped
    env.reset(seed=5)
    _, r, *_ = env.step(C.HOLD)
    assert r == pytest.approx(0.01)
    _, r, *_ = env.step(C.ASCEND)                      # action change
    assert r == pytest.approx(0.01 - 0.002)
    total_cleared, rewards = 0, []
    while True:                                        # dive into the ground
        _, r, term, trunc, info = env.step(C.DESCEND)
        rewards.append(r)
        if term:
            break
    assert info["crash_cause"] == "ground"
    assert rewards[-1] == pytest.approx(-5.0)


def test_clearing_reward():
    from drone_env.sim import Simulator
    env = gym.make("DroneDash-v0", level="L1", max_frames=None).unwrapped
    env.reset(seed=7)
    sim = Simulator("L1", 7, generate=False)            # controlled world: 3 low buildings
    sim.y = 6.0
    for x in (10.0, 20.0, 30.0):
        sim.place_static(C.BUILDING, sim.x + x, sim.x + x + 2.0, 0.0, 4.0)
    env.sim = sim
    got = 0
    for _ in range(300):
        _, r, term, _, info = env.step(C.HOLD)
        assert not term
        if r > 0.5:
            assert r == pytest.approx(1.01)
            got += 1
    assert got == 3 and info["cleared"] == 3


def test_action_descriptors():
    env = gym.make("DroneDash-v0").unwrapped
    d = env.action_descriptors()
    assert d.shape == (5, 3) and d.dtype == np.float32
    assert d[C.HOLD, 0] == 0 and d[C.ASCEND, 0] > 0 > d[C.DESCEND, 0]
    assert d[C.ASCEND, 0] == pytest.approx(-d[C.DESCEND, 0])
    assert d[C.JUMP, 0] > d[C.ASCEND, 0] and d[C.DUCK, 0] < d[C.DESCEND, 0]
    assert d[C.DUCK, 0] == pytest.approx(-d[C.JUMP, 0])        # the dive mirrors the jump (v1.7)
    assert d[C.DUCK, 1] == pytest.approx(0.6) and (d[:4, 1] == 1).all()
    assert d[C.JUMP, 2] == pytest.approx(0.3) and d[C.DUCK, 2] == pytest.approx(0.3)


def test_rgb_render():
    env = gym.make("DroneDash-v0", level="L5", render_mode="rgb_array")
    env.reset(seed=3)
    env.step(0)
    img = env.render()
    assert img.shape == (400, 1040, 3) and img.dtype == np.uint8 and img.std() > 0     # 26 m x 40 px (v2.3)
    env.close()
