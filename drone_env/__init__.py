"""DroneDash: a Chrome-Dino-style side-scroller for a self-flying drone (SPEC v1.3).

Importing this package registers ``DroneDash-v0``:

    import gymnasium, drone_env
    env = gymnasium.make("DroneDash-v0", level="L3", render_mode=None, max_frames=10_000)
"""
from gymnasium.envs.registration import register, registry

from .constants import ACTION_NAMES, LEVELS, SPEC_VERSION, config_hash  # noqa: F401

if "DroneDash-v0" not in registry:
    register(id="DroneDash-v0", entry_point="drone_env.env:DroneDashEnv")

__all__ = ["ACTION_NAMES", "LEVELS", "SPEC_VERSION", "config_hash"]
