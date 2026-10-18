import gymnasium as gym
import numpy as np

import drone_env  # noqa: F401
from drone_env import constants as C

from .helpers import empty_sim

ALL = [1, 1, 1, 1, 1]
BURST = [1, 0, 0, 0, 0]
COOLDOWN = [1, 1, 1, 0, 0]          # JUMP and DUCK share one cooldown (v1.7)


def test_jump_burst_and_cooldown():
    s = empty_sim()
    assert s.action_mask().tolist() == ALL
    s.step(C.JUMP)
    for i in range(1, 9):                       # frames 2..9 of the 9-frame burst
        assert s.action_mask().tolist() == BURST, i
        s.step(C.HOLD)
    for i in range(9, 30):                      # rest of the 1 s cooldown
        assert s.action_mask().tolist() == COOLDOWN, i
        s.step(C.HOLD)
    assert s.action_mask().tolist() == ALL      # 30 frames after the jump started


def test_dive_burst_tuck_and_shared_cooldown():
    s = empty_sim()
    s.step(C.DUCK)
    assert s.vy == C.DUCK_V or s.vy < -5.9      # vy = -6 at once
    for i in range(1, 9):                       # frames 2..9 of the 9-frame dive
        assert s.action_mask().tolist() == BURST, i
        assert s.tucked
        s.step(C.HOLD)
    assert not s.tucked
    for i in range(9, 30):                      # shared cooldown: no JUMP, no DUCK
        assert s.action_mask().tolist() == COOLDOWN, i
        s.step(C.HOLD)
    assert s.action_mask().tolist() == ALL


def test_jump_blocks_duck_and_duck_blocks_jump():
    for first, second in ((C.JUMP, C.DUCK), (C.DUCK, C.JUMP)):
        s = empty_sim()
        s.step(first)
        for _ in range(12):
            s.step(C.HOLD)
        s.step(second)                          # inside the shared cooldown: becomes HOLD
        assert s.last_action == C.HOLD and s.masked_count == 1 and s.burst == 0


def test_masked_action_becomes_hold_and_is_counted():
    s = empty_sim()
    s.step(C.JUMP)
    s.step(C.ASCEND)                            # masked during the burst
    assert s.last_action == C.HOLD and s.masked_count == 1
    for _ in range(10):
        s.step(C.HOLD)
    s.step(C.JUMP)                              # masked during cooldown
    assert s.last_action == C.HOLD and s.masked_count == 2 and s.burst == 0
    s2 = empty_sim()
    s2.step(C.DUCK)
    s2.step(C.JUMP)
    assert s2.last_action == C.HOLD and s2.masked_count == 1


def test_env_counts_masked_actions_in_info():
    env = gym.make("DroneDash-v0", level="L1").unwrapped
    obs, info = env.reset(seed=3)
    assert info["masked_action_used"] == 0
    obs, *_ , info = env.step(C.JUMP)
    assert obs["action_mask"].tolist() == BURST
    obs, *_ , info = env.step(C.DUCK)
    assert info["masked_action_used"] == 1
    assert obs["self"][6 + C.HOLD] == 1.0       # last action one-hot shows the HOLD it became
