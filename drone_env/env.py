"""Gymnasium environment ``DroneDash-v0`` (SPEC §5–7)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import constants as C
from .sim import Simulator

REWARD_ALIVE = 0.01
REWARD_CLEARED = 1.0
REWARD_CRASH = -5.0
REWARD_CHANGE = -0.002


def observation_space(spec: tuple = C.SENSOR_SPEC) -> spaces.Dict:
    """For a sensor spec: v2.3 rows have 13 values and rays reach 20/16; older pilots' 8 and 1."""
    big = np.float32(1e3)
    return spaces.Dict({
        "self": spaces.Box(-big, big, (11,), np.float32),
        "rays": spaces.Box(0.0, C.sensor_range(spec) / C.NORM_RANGE, (C.N_RAYS,), np.float32),
        "hazards": spaces.Box(-big, big, (C.MAX_HAZARDS, C.hazard_cols(spec)), np.float32),
        "hazard_type": spaces.Box(0, C.N_HAZARD_TYPES - 1, (C.MAX_HAZARDS,), np.int64),   # v2.0: 8-12 new kinds
        "hazard_mask": spaces.Box(0, 1, (C.MAX_HAZARDS,), np.int8),
        "action_mask": spaces.Box(0, 1, (C.N_ACTIONS,), np.int8),
    })


def action_descriptors() -> np.ndarray:
    """(5, 3) float32 per action, computed from the physics constants (SPEC §4):

    0. height gained 0.5 s after starting from a hover (action held where the mask allows,
       HOLD otherwise), divided by 3
    1. hitbox height scale
    2. maneuver duration in seconds (one frame for HOLD / ASCEND / DESCEND)
    """
    out = np.zeros((C.N_ACTIONS, 3), dtype=np.float32)
    frames = int(round(0.5 * C.FPS))
    for a in range(C.N_ACTIONS):
        s = Simulator("L1", 0, generate=False)
        s.y = 5.0
        y0 = s.y
        for _ in range(frames):
            s.step(a if a in s.available_actions() else C.HOLD)
        out[a, 0] = (s.y - y0) / 3.0
    out[:, 1] = 1.0
    out[C.DUCK, 1] = C.TUCK_SCALE
    out[:, 2] = C.DT
    out[C.JUMP, 2] = C.JUMP_BURST * C.DT
    out[C.DUCK, 2] = C.DUCK_FRAMES * C.DT
    return out


_ACTION_DESC = action_descriptors()


class DroneDashEnv(gym.Env):
    """DroneDash. ``env.unwrapped.sim`` is the Simulator (for planners and the viewer only)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": C.FPS}

    def __init__(self, level: str = "L1", render_mode: str | None = None, max_frames: int | None = 10_000):
        if level not in C.LEVELS:
            raise ValueError(f"unknown level {level!r}; choose one of {C.LEVELS}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unknown render_mode {render_mode!r}")
        self.level = level
        self.render_mode = render_mode
        self.max_frames = max_frames
        self.observation_space = observation_space()
        self.action_space = spaces.Discrete(C.N_ACTIONS)
        self.sim: Simulator | None = None
        self.sim_seed: int | None = None
        self._last_obs = None
        self._renderer = None
        self._sensor_spec = C.SENSOR_SPEC

    @property
    def sensor_spec(self) -> tuple:
        """Sensor spec the observations follow (``agent.sensor_spec``: pilots made before v2.3 get
        theirs, SPEC v2.2/v2.3). Applies from the next reset."""
        return self._sensor_spec

    @sensor_spec.setter
    def sensor_spec(self, spec) -> None:
        self._sensor_spec = tuple(spec)
        self.observation_space = observation_space(self._sensor_spec)

    # ---------------------------------------------------------------------------------
    def action_descriptors(self) -> np.ndarray:
        return _ACTION_DESC.copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.sim_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.sim = Simulator(self.level, self.sim_seed)
        self.sim.sensor_spec = self._sensor_spec
        obs = self.sim.observe()
        self._last_obs = obs
        info = self._info(obs)
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action):
        if self.sim is None:
            raise RuntimeError("call reset() before step()")
        sim = self.sim
        prev = sim.last_action
        crashed = sim.step(int(action))
        reward = REWARD_CRASH if crashed else REWARD_ALIVE
        reward += REWARD_CLEARED * sim.cleared_step
        if sim.last_action != prev:
            reward += REWARD_CHANGE
        terminated = bool(crashed)
        truncated = (not terminated) and self.max_frames is not None and sim.frame >= self.max_frames
        obs = sim.observe()
        self._last_obs = obs
        info = self._info(obs)
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, bool(truncated), info

    def _info(self, obs) -> dict:
        s = self.sim
        return {
            "frame": s.frame,
            "distance_m": s.distance_m,
            "speed": s.speed,
            "level": self.level,
            "level_now": s.level_now,
            "seed": self.sim_seed,
            "crash_cause": s.crash_cause,
            "gust_active": s.gust_active,
            "surge_active": s.surge_active,
            "fog_active": s.fog_active,
            "hazards_visible": int(obs["hazard_mask"].sum()),
            "masked_action_used": s.masked_count,
            "action": s.last_action,
            "cleared": s.cleared_total,
        }

    # ---------------------------------------------------------------------------------
    def render(self):
        if self.render_mode is None or self.sim is None:
            return None
        if self._renderer is None:
            from .render import Renderer
            self._renderer = Renderer(window=self.render_mode == "human")
        if self.render_mode == "rgb_array":
            return self._renderer.rgb_array(self.sim)
        self._renderer.show(self.sim)
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
