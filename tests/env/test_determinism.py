import gymnasium as gym
import pytest

import drone_env  # noqa: F401
from app.agents_base import RuleAgent
from drone_env import constants as C
from drone_env.sim import Simulator

from .helpers import scripted


def record_script(level, seed, frames=2000):
    """Record the rule agent's actions on the env: a fixed script to replay on bare sims."""
    env = gym.make("DroneDash-v0", level=level, max_frames=frames).unwrapped
    agent = RuleAgent()
    agent.reset(env.action_descriptors())
    obs, _ = env.reset(seed=seed)
    script = []
    for _ in range(frames):
        a = agent.act(obs).action
        script.append(a)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    # keep the script 2,000 frames long; after a crash, step() is a no-op
    return script + [C.HOLD] * (frames - len(script))


def replay(level, seed, script):
    s = Simulator(level, seed)
    for a in script:
        s.step(a)
    return s


@pytest.mark.parametrize("level", C.LEVELS)
def test_same_seed_same_hash_different_seed_different_hash(level):
    script = record_script(level, 11)
    a, b, c = replay(level, 11, script), replay(level, 11, script), replay(level, 12, script)
    assert a.state_hash() == b.state_hash()
    assert a.state_hash() != c.state_hash()


def _env_vs_sim(level, seed, frames=2000):
    """Fly the env with the rule agent (observing every frame) and replay the same actions on a
    bare Simulator that never observes. Return the number of fog frames seen."""
    env = gym.make("DroneDash-v0", level=level, max_frames=None).unwrapped
    agent = RuleAgent()
    agent.reset(env.action_descriptors())
    obs, _ = env.reset(seed=seed)
    bare = Simulator(level, seed)
    fog_frames = 0
    for _ in range(frames):
        a = agent.act(obs).action
        obs, _, term, _, info = env.step(a)
        bare.step(a)
        assert env.sim.state_tuple() == bare.state_tuple()
        fog_frames += int(info["fog_active"])
        if term:
            break
    return fog_frames


def test_env_and_bare_simulator_stay_identical_through_fog():
    fog_total = 0
    for seed in (300_000, 300_001, 300_002, 300_003):
        fog_total += _env_vs_sim("stress", seed)
        if fog_total >= 60:
            break
    assert fog_total >= 60, "no fog bank was flown through; pick other seeds"


def test_observe_does_not_change_world():
    s = Simulator("stress", 300_001)
    from drone_env import events
    events.start(s, events.FOG, 5.0)
    for f in range(300):
        h = s.state_hash()
        s.observe()
        s.observe()
        assert s.state_hash() == h
        s.step(scripted(s, f))


def test_reset_seed_reproduces_world():
    env = gym.make("DroneDash-v0", level="L5").unwrapped
    o1, _ = env.reset(seed=42)
    h1 = env.sim.state_hash()
    env.reset(seed=43)
    o2, _ = env.reset(seed=42)
    assert env.sim.state_hash() == h1
    for k in o1:
        assert (o1[k] == o2[k]).all()
