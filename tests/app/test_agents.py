import gymnasium as gym
import numpy as np
import pytest

import drone_env  # noqa: F401
from app.agents_base import REGISTRY, Agent, Decision, load_agent


def test_builtin_agents_load_and_follow_the_protocol():
    for spec in ("rule", "random", "random:3"):
        a = load_agent(spec)
        assert isinstance(a, Agent) and a.name in ("rule", "random")


def test_unknown_and_onnx_specs():
    with pytest.raises(ValueError, match="unknown agent"):
        load_agent("nope")
    if "onnx-arc" not in REGISTRY:
        with pytest.raises(ValueError, match="onnx agents not installed yet"):
            load_agent("onnx-arc:artifacts/arc.onnx")
        with pytest.raises(ValueError, match="onnx agents not installed yet"):
            load_agent("onnx-mlp:artifacts/mlp.onnx")


def test_random_agent_uses_only_available_actions():
    env = gym.make("DroneDash-v0", level="L1").unwrapped
    a = load_agent("random:1")
    a.reset(env.action_descriptors())
    obs, _ = env.reset(seed=1)
    for _ in range(300):
        d = a.act(obs)
        assert isinstance(d, Decision) and obs["action_mask"][d.action] == 1
        obs, _, term, _, info = env.step(d.action)
        if term:
            break
    assert info["masked_action_used"] == 0
    b1, b2 = load_agent("random:5"), load_agent("random:5")
    assert [b1.act(obs).action for _ in range(20)] == [b2.act(obs).action for _ in range(20)]


def test_rule_agent_sees_only_the_observation():
    env = gym.make("DroneDash-v0", level="L3").unwrapped
    a = load_agent("rule")
    a.reset(env.action_descriptors())
    obs, _ = env.reset(seed=100_000)
    frozen = {k: v.copy() for k, v in obs.items()}
    for v in frozen.values():
        v.setflags(write=False)
    d = a.act(frozen)                       # works on a plain read-only dict
    assert 0 <= d.action < 5
    assert d.attended is None or obs["hazard_mask"][d.attended] == 1


@pytest.mark.parametrize("plane_y, room, expect", [
    (5.0, "both", {3, 4}),           # open sky: either burst
    (5.0, "low_ceiling", {4}),       # storm dip above: dive under the plane
    (5.0, "tall_floor", {3}),        # tall building below: jump over it
])
def test_rule_agent_bursts_over_or_under_a_plane_about_to_hit(plane_y, room, expect):
    """SPEC v1.7: a plane at the drone's height < 0.45 s away -> JUMP or DUCK by the room."""
    from drone_env import constants as C
    from drone_env import events
    from drone_env.sim import Simulator
    sim = Simulator("L5", 0, generate=False)
    events.make_calm(sim)
    sim.y = plane_y
    if room == "low_ceiling":
        sim.place_static(C.STORM, sim.x + 2.0, sim.x + 14.0, 7.2, C.CEILING)
    elif room == "tall_floor":
        sim.place_static(C.BUILDING, sim.x + 2.0, sim.x + 6.0, 0.0, 2.6)
    sim.place_mover(C.PLANE, sim.x + C.HALF_W + 8.0, plane_y, 5.0, 1.0, s=8.0)   # 8 m / 24 m/s
    a = load_agent("rule")
    a.reset(gym.make("DroneDash-v0").unwrapped.action_descriptors())
    d = a.act(sim.observe())
    assert d.action in expect, (room, d)
