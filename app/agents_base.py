"""Agent interface (SPEC §8), the agent registry and the two built-in agents.

An agent sees only the observation dict (SPEC §5). It never gets the simulator.
No pygame imports here: this module is used headless (tests, bench, evaluation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

HOLD, ASCEND, DESCEND, JUMP, DUCK = 0, 1, 2, 3, 4
BUILDING, MOUNTAIN, BIRD, TRAFFIC, STORM, DOWNDRAFT, PLANE = 1, 2, 3, 4, 5, 6, 7
TREE, POLE, POWERLINE, KITE, CRANE = 8, 9, 10, 11, 12           # SPEC v2.0
MOVERS = (BIRD, TRAFFIC, PLANE)
GROUND = (BUILDING, MOUNTAIN, TREE, POLE, KITE)   # blocked from the ground up: pass over
# power lines and cranes block a band in mid air: like a level flyer, they bound whichever side the drone is on
HALF_H = 0.175
BURST_ROOM = 3.6        # a JUMP or a DUCK (dive) moves the drone ~3.3 m before it can stop


@dataclass
class Decision:
    action: int
    probs: np.ndarray | None = None      # (5,) after temperature scaling
    p_safe: np.ndarray | None = None     # (5,) P(still flying after 1 s | action)
    threat: int | None = None            # 0..4
    attended: int | None = None          # row index into obs["hazards"], for the HUD outline
    hesitating: bool = False             # max(probs) < tau. Display and logging only.


@runtime_checkable
class Agent(Protocol):
    name: str

    def reset(self, action_desc: np.ndarray) -> None: ...

    def act(self, obs: dict) -> Decision: ...


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
AgentFactory = Callable[[str], Agent]       # receives the text after "name:" (may be "")
REGISTRY: dict[str, AgentFactory] = {}


def register_agent(name: str, factory: AgentFactory) -> None:
    """Session 2's ``agents`` package calls this for "onnx-arc" and "onnx-mlp"."""
    REGISTRY[name] = factory


def load_agent(spec: str) -> Agent:
    """Build an agent from a spec such as ``rule``, ``random``, ``random:7`` or
    ``onnx-arc:artifacts/arc.onnx``."""
    name, _, arg = spec.partition(":")
    if name not in REGISTRY:
        try:
            import agents  # noqa: F401  (session 2 registers its specs on import)
        except ImportError:
            pass
    if name not in REGISTRY:
        if name.startswith("onnx"):
            raise ValueError(f"agent {spec!r}: onnx agents not installed yet "
                             "(session 2 adds them in agents/onnx_agents.py)")
        raise ValueError(f"unknown agent {spec!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](arg)


# --------------------------------------------------------------------------------------
# Built-in agents
# --------------------------------------------------------------------------------------
class RandomAgent:
    """Uniform over the available actions."""

    def __init__(self, seed: int | None = None):
        self.name = "random"
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, action_desc: np.ndarray) -> None:
        pass

    def act(self, obs: dict) -> Decision:
        avail = np.flatnonzero(obs["action_mask"])
        return Decision(action=int(self.rng.choice(avail)))


class RuleAgent:
    """Simple hand-written autopilot for testing the game (not a learned reflex, no training).

    * Reflex: a plane or a traffic drone/balloon about to hit the drone at its height (contact
      in under 0.45 s) -> JUMP over it or DUCK (dive) under it, whichever side has room for the
      ~3.3 m a burst moves the drone.
    * Look at the nearest hazard ahead. If it will overlap the drone's altitude band within
      ``near_m`` metres, steer toward the larger free gap at that hazard: ascend or descend, or
      JUMP / DUCK when the gap is far above / below and contact is close.
    * Otherwise drift toward the middle of the free corridor over the next ``look_s`` seconds.
    """

    def __init__(self, near_m: float = 6.0, look_s: float = 1.6, margin: float = 0.45):
        self.name = "rule"
        self.near_m = near_m
        self.look_s = look_s
        self.margin = margin

    def reset(self, action_desc: np.ndarray) -> None:
        self.jump_gain = float(action_desc[JUMP, 0]) * 3.0 if action_desc is not None else 2.7

    @staticmethod
    def _rows(obs):
        hz = obs["hazards"]
        lo, hi = (11, 12) if hz.shape[1] >= 13 else (1, 2)    # v2.3 rows: the tracker's predicted band
        out = []
        for i in np.flatnonzero(obs["hazard_mask"]):
            out.append((i, int(obs["hazard_type"][i]), float(hz[i, 0]) * 16, float(hz[i, lo]) * 10,
                        float(hz[i, hi]) * 10, float(hz[i, 3]) * 8, -float(hz[i, 4]) * 16,
                        float(hz[i, 5]) * 8))
        return out   # (row, kind, gap_m, y_bottom, y_top, width, closing_speed, vy)

    def _corridor(self, rows, y, upto_m, exclude=None):
        """Floor and ceiling of the free space over the hazards within upto_m metres."""
        floor, ceil = 0.0, 10.0
        for i, k, gap, y0, y1, w, close, vy in rows:
            if gap > upto_m or k == DOWNDRAFT or i == exclude:
                continue
            if k in GROUND:
                floor = max(floor, y1)
            elif k == STORM:
                ceil = min(ceil, y0)
            else:   # mover: it bounds whichever side the drone is on
                pad = 1.6 if k == BIRD else 0.25
                if (y0 + y1) / 2 >= y:
                    ceil = min(ceil, y0 - pad)
                else:
                    floor = max(floor, y1 + pad)
        return floor, ceil

    def act(self, obs: dict) -> Decision:
        s = obs["self"]
        y, vy = float(s[0]) * 10, float(s[1]) * 8
        speed = float(s[4]) * 8 + 8          # SPEC v1.7 §5: self[4] = (v - 8) / 8
        mask = obs["action_mask"].astype(bool)
        rows = self._rows(obs)
        look = max(self.near_m, speed * self.look_s)
        ahead = [r for r in rows if r[1] != DOWNDRAFT]
        nearest = min(ahead, key=lambda r: (r[2], r[0])) if ahead else None
        attended = nearest[0] if nearest is not None else None
        band0, band1 = y - HALF_H - self.margin, y + HALF_H + self.margin

        # reflex: something flying is about to hit us at our height -> burst over or under it
        if mask[JUMP] or mask[DUCK]:
            for i, k, gap, y0, y1, w, close, hvy in rows:
                if k not in (TRAFFIC, PLANE):     # level flyers: where they are is where they'll be
                    continue
                t_hit = gap / max(close, 0.5)
                if t_hit > 0.45 or y0 - 0.15 > y + HALF_H or y1 + 0.15 < y - HALF_H:
                    continue
                floor, ceil = self._corridor(rows, y, gap + w + speed * 0.6, exclude=i)
                up_ok = mask[JUMP] and ceil - y > BURST_ROOM
                dn_ok = mask[DUCK] and y - floor > BURST_ROOM
                if up_ok and (not dn_ok or ceil - y1 >= y0 - floor):
                    return Decision(action=JUMP, attended=int(i))
                if dn_ok:
                    return Decision(action=DUCK, attended=int(i))

        target = None
        jump_ok = dive_ok = False
        if nearest is not None:
            i, k, gap, y0, y1, w, close, hvy = nearest
            # where will a moving hazard be when we get there?
            t_hit = gap / max(close, 0.5)
            if k == BIRD:      # a bird can be anywhere on its sine path (amplitude <= 1.5 m)
                y0, y1 = y0 - 1.5 - 0.1, y1 + 1.5 + 0.1
            elif k in (TRAFFIC, PLANE):
                y0, y1 = y0 - 0.25, y1 + 0.25
            if gap <= look and y0 <= band1 and y1 >= band0:
                # free gaps at this hazard, bounded by other hazards over the same stretch
                floor, ceil = 0.0, 10.0
                for j, k2, g2, a0, a1, w2, c2, v2 in rows:
                    if j == i or k2 == DOWNDRAFT or g2 > gap + w + 1.0 or g2 + w2 < gap - 1.0:
                        continue
                    if k2 in GROUND:
                        floor = max(floor, a1)
                    elif k2 == STORM:
                        ceil = min(ceil, a0)
                    else:
                        pad = 1.6 if k2 == BIRD else 0.25
                        if (a0 + a1) / 2 > (y0 + y1) / 2:
                            ceil = min(ceil, a0 - pad)
                        else:
                            floor = max(floor, a1 + pad)
                below = (y0 - floor) if k not in GROUND else 0.0
                above = (ceil - y1) if k != STORM else 0.0
                if above >= below:
                    target = y1 + min(above / 2, 1.2)
                    jump_ok = ceil - y > BURST_ROOM
                else:
                    target = y0 - min(below / 2, 1.2)
                    dive_ok = y - floor > BURST_ROOM
                urgency = t_hit
        if target is None:
            floor, ceil = self._corridor(rows, y, look)
            target = (floor + ceil) / 2 if ceil - floor < 6.0 else min(max(y, floor + 2.0), ceil - 2.0)
            urgency = None
        target = min(max(target, 0.6), 9.4)

        err = target - (y + vy * 0.2)
        if urgency is not None and jump_ok and mask[JUMP] and err > 1.3 and urgency < 0.8:
            a = JUMP
        elif urgency is not None and dive_ok and mask[DUCK] and err < -1.3 and urgency < 0.8:
            a = DUCK
        elif err > 0.15:
            a = ASCEND
        elif err < -0.15:
            a = DESCEND
        else:
            a = HOLD
        if not mask[a]:
            a = DESCEND if (a == DESCEND and mask[DESCEND]) else HOLD
        return Decision(action=int(a), attended=None if attended is None else int(attended))


def _random_factory(arg: str) -> Agent:
    return RandomAgent(int(arg) if arg else None)


def _rule_factory(arg: str) -> Agent:
    return RuleAgent()


register_agent("random", _random_factory)
register_agent("rule", _rule_factory)
