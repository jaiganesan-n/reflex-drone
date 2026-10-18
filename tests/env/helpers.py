import numpy as np

from drone_env import constants as C
from drone_env.sim import Simulator


def scripted(sim: Simulator, frame: int) -> int:
    """Deterministic scripted pilot (uses only the sim's own state; tests only)."""
    if frame % 97 == 40:
        return C.JUMP
    if frame % 131 == 70:
        return C.DUCK
    return C.ASCEND if sim.y < 4.5 else (C.DESCEND if sim.y > 5.5 else C.HOLD)


def empty_sim(y: float = 5.0, level: str = "L1") -> Simulator:
    s = Simulator(level, 0, generate=False)
    s.y = y
    return s


def reference_climb(actions, y0=5.0, vy0=0.0):
    """Independent integrator of SPEC §2 vertical dynamics without wind (v1.7: DUCK = dive)."""
    y, vy = y0, vy0
    burst = cool = 0
    burst_v = 0.0
    for a in actions:
        if burst > 0:
            vt = burst_v
        elif a == C.ASCEND:
            vt = 3.0
        elif a == C.DESCEND:
            vt = -3.0
        elif a == C.JUMP and cool == 0:
            vy, burst, cool, vt, burst_v = 6.0, 9, 30, 6.0, 6.0
        elif a == C.DUCK and cool == 0:
            vy, burst, cool, vt, burst_v = -6.0, 9, 30, -6.0, -6.0
        else:
            vt = 0.0
        acc = float(np.clip(8.0 * (vt - vy), -12.0, 12.0))
        vy += acc * C.DT
        y += vy * C.DT
        burst, cool = max(burst - 1, 0), max(cool - 1, 0)
    return y, vy
