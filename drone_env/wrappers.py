"""Flat view for the MLP baseline (SPEC §5).

``flatten_obs`` is the single source of truth for the 40-value flat frame. Session 2's dataset
loader and MLP agent import it; the ``[previous, current]`` stacking rule lives here too.
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

from . import constants as C

FLAT_DIM = 40
N_NEAREST = 3
_PAD = np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)   # missing hazard: dx = 1, rest 0
_AIRBORNE = np.array(C.AIRBORNE_KINDS, dtype=np.int64)        # bird, traffic, plane; v2.0 power line, crane


def flatten_obs(self_, rays, hazards, hazard_type, hazard_mask) -> np.ndarray:
    """Pure, batched flattening of observation arrays to (N, 40) float32.

    Accepts float16 or float32 inputs with a leading batch dimension (a single unbatched frame
    is also accepted and returned as (1, 40)). Per frame:

    * ``self[:6]``                                   6 values
    * ``rays``                                       13 values
    * the 3 valid hazards with the smallest dx, each as
      (dx, y_top, y_bottom, width, vy, airborne = constants.AIRBORNE_KINDS, moving)   3 x 7 values
      missing ones padded with dx = 1, rest 0.

    The choice of the 3 hazards does not depend on row order: ties in dx are broken by the
    other features, so shuffling hazard rows never changes the result.
    """
    s = np.asarray(self_, dtype=np.float32)
    single = s.ndim == 1
    r = np.asarray(rays, dtype=np.float32)
    hz = np.asarray(hazards, dtype=np.float32)
    ht = np.asarray(hazard_type).astype(np.int64)
    hm = np.asarray(hazard_mask).astype(bool)
    if single:
        s, r, hz, ht, hm = s[None], r[None], hz[None], ht[None], hm[None]
    n = s.shape[0]
    dx_key = np.where(hm, hz[..., 0], np.inf)
    # np.lexsort: the LAST key is the primary key.
    keys = (hz[..., 7], hz[..., 5], hz[..., 4], hz[..., 3], hz[..., 6], hz[..., 1], hz[..., 2],
            ht.astype(np.float32), dx_key)
    keys = tuple(np.where(hm, k, 0.0) for k in keys[:-1]) + (dx_key,)
    order = np.lexsort(keys, axis=-1)[:, :N_NEAREST]                        # (N, 3)
    sel = np.take_along_axis(hz, order[..., None], axis=1)                  # (N, 3, 8)
    sel_t = np.take_along_axis(ht, order, axis=1)
    sel_m = np.take_along_axis(hm, order, axis=1)
    airborne = np.isin(sel_t, _AIRBORNE).astype(np.float32)
    feats = np.stack([sel[..., 0], sel[..., 2], sel[..., 1], sel[..., 3], sel[..., 5], airborne,
                      sel[..., 7]], axis=-1)                                # (N, 3, 7)
    feats = np.where(sel_m[..., None], feats, _PAD)
    out = np.concatenate([s[:, :6], r, feats.reshape(n, N_NEAREST * 7)], axis=1)
    return out.astype(np.float32, copy=False)


def flatten_dict(obs: dict) -> np.ndarray:
    """flatten_obs for one observation dict (unbatched) -> (40,) float32."""
    return flatten_obs(obs["self"], obs["rays"], obs["hazards"], obs["hazard_type"], obs["hazard_mask"])[0]


def stack_frames(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """The SPEC stacking rule: concatenate [previous, current] on the last axis (40 -> 80)."""
    return np.concatenate([prev, cur], axis=-1)


class FlatHistory:
    """[previous, current] buffer for agents: reset() at episode start, push(obs) every frame."""

    def __init__(self, n: int = 2):
        self.n = n
        self.buf: deque = deque(maxlen=n)

    def reset(self) -> None:
        self.buf.clear()

    def push(self, obs: dict) -> np.ndarray:
        f = flatten_dict(obs)
        if not self.buf:
            for _ in range(self.n - 1):
                self.buf.append(f)        # first frame duplicated
        self.buf.append(f)
        return np.concatenate(list(self.buf), axis=-1)


class FlattenObs(gym.ObservationWrapper):
    """Dict observation -> (40,) float32 via flatten_obs."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (FLAT_DIM,), np.float32)

    def observation(self, obs):
        return flatten_dict(obs)


class FrameStack(gym.Wrapper):
    """Concatenate the last n flat frames, oldest first ([previous, current] for n = 2).

    At reset the first frame is duplicated. A Dict-observation env is flattened first.
    """

    def __init__(self, env: gym.Env, n: int = 2):
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObs(env)
        super().__init__(env)
        self.n = n
        d = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (d * n,), np.float32)
        self._frames: deque = deque(maxlen=n)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.n):
            self._frames.append(obs)
        return np.concatenate(list(self._frames)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return np.concatenate(list(self._frames)), reward, terminated, truncated, info
