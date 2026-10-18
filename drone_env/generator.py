"""Level schedule, hazard objects and the layout generator with its passability check.

World coordinates: x grows to the right (the direction of flight), the drone centre sits at
world x = distance_m + DRONE_X. Static hazards never move in world x. Moving hazards (birds,
traffic) are created *dormant* with the world x where the drone is meant to meet them
(``xe``); they activate when the drone is LOOKAHEAD metres away and from then on follow a
closed-form path, so no hazard object is ever mutated after creation. That keeps
``Simulator.clone()`` cheap: hazard lists are copied, hazard objects are shared.

Passability (SPEC §3): the generator keeps a reachable altitude set R(x) for the drone centre,
sweeping x in CHECK_DX steps. Between samples R grows by the climb/descent rate (3 m/s) times
the time the nominal speed allows, then is cut to the free corridors (vertical gaps >= 1.2 m,
shrunk by the drone's half height). A candidate obstacle is accepted only if R stays non-empty
over every footprint known so far; otherwise it is resampled (MAX_TRIES), and if nothing fits an
empty gap is inserted instead (always passable).

Reflex checks (SPEC v1.7): the "consecutive corridors" check (from the middle +/- 1 m of the
previous obstacle's tightest corridor), a first-sight check for moving hazards, and for planes a
real-physics escape check (``plane_escape_ok``) plus spacing for the shared JUMP/DUCK cooldown.
Endless past L5 ("L5+") keeps the speed and shrinks the gaps (``Schedule.density_at``).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import constants as C

NEVER = 1 << 60


# ------------------------------------------------------------------------------------
# Levels
# ------------------------------------------------------------------------------------
class LevelSpec:
    __slots__ = ("name", "speed", "kinds", "cum", "gap_lo", "gap_hi", "combo_p", "gust_every", "gust_lo",
                 "gust_hi", "surge_every", "fog_every", "exclusive", "speed_mult")

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.speed = float(cfg["speed"])
        mix = cfg["mix"]
        kinds, weights = [], []
        for k, w in mix.items():
            if w > 0:
                kinds.append(C.HAZARD_IDS[k])
                weights.append(float(w))
        tot = sum(weights)
        cum, acc = [], 0.0
        for w in weights:
            acc += w / tot
            cum.append(acc)
        cum[-1] = 1.0
        self.kinds = tuple(kinds)
        self.cum = tuple(cum)
        self.gap_lo = max(float(cfg["gap_s"][0]), C.MIN_GAP_S)
        self.gap_hi = max(float(cfg["gap_s"][1]), self.gap_lo)
        self.combo_p = float(cfg.get("combo_p", 0.0))
        self.gust_every = float(cfg["gust_every_s"])
        self.gust_lo, self.gust_hi = (float(v) for v in cfg["gust_accel"])
        self.surge_every = float(cfg["surge_every_s"])
        self.fog_every = float(cfg["fog_every_s"])
        self.exclusive = bool(cfg["exclusive"])
        self.speed_mult = float(cfg.get("speed_mult", 1.0))


@lru_cache(maxsize=None)
def level_spec(name: str) -> LevelSpec:
    return LevelSpec(name, C.LEVELS_CFG["levels"][name])


class Schedule:
    """Maps distance flown to the nominal (surge-free) scroll speed and the active level.

    A fixed level flies at its own speed. In ``endless`` the levels start at equal distances over
    ``ENDLESS_RAMP_M`` metres (level i at i * ramp / (n - 1)), and the speed moves linearly from each
    level's speed to the next one's between their starts (SPEC v2.3: 10 -> 12 m/s over L1, then
    12 m/s; with equally spaced speeds, as before v2.3, this is the old 8 -> 16 m/s straight ramp).
    """

    __slots__ = ("level", "specs", "multi", "ramp_m", "v0", "v1", "starts", "speeds")

    def __init__(self, level: str):
        if level not in C.LEVELS:
            raise ValueError(f"unknown level {level!r}; choose one of {C.LEVELS}")
        self.level = level
        if level == "endless":
            self.specs = tuple(level_spec(n) for n in C.ENDLESS_SEQUENCE)
            self.multi = True
            self.ramp_m = C.ENDLESS_RAMP_M
            n = len(self.specs)
            self.starts = tuple(self.ramp_m * i / (n - 1) for i in range(n))
            self.speeds = tuple(sp.speed for sp in self.specs)
        else:
            self.specs = (level_spec(level),)
            self.multi = False
            self.ramp_m = math.inf
            self.starts = (0.0,)
            self.speeds = (self.specs[0].speed,)
        self.v0, self.v1 = self.speeds[0], self.speeds[-1]

    def speed_at(self, d: float) -> float:
        if not self.multi:
            sp = self.specs[0]
            return sp.speed * sp.speed_mult
        st, vs = self.starts, self.speeds
        if d <= 0.0:
            return vs[0]
        for i in range(1, len(st)):
            if d < st[i]:
                return vs[i - 1] + (vs[i] - vs[i - 1]) * (d - st[i - 1]) / (st[i] - st[i - 1])
        return vs[-1]

    def index_at(self, d: float) -> int:
        if not self.multi:
            return 0
        i = 0
        for k, s0 in enumerate(self.starts):
            if d + 1e-9 >= s0:
                i = k
        return i

    def spec_at(self, d: float) -> LevelSpec:
        return self.specs[self.index_at(d)]

    def past_l5_km(self, d: float) -> float:
        """Endless only: km flown past the start of the last level ("L5+"), else 0."""
        if not self.multi or d <= self.ramp_m:
            return 0.0
        return (d - self.ramp_m) / 1000.0

    def density_at(self, d: float, spec: "LevelSpec"):
        """(gap_lo, gap_hi, combo_p) at distance d: the level's own values, made denser in L5+."""
        km = self.past_l5_km(d)
        if km <= 0.0:
            return spec.gap_lo, spec.gap_hi, spec.combo_p
        f = max(1.0 - C.L5P_GAP_SHRINK * km, C.L5P_GAP_FLOOR_FRAC)
        lo = max(spec.gap_lo * f, C.L5P_MIN_GAP_S)
        hi = max(spec.gap_hi * f, lo + 0.2)
        return lo, hi, min(spec.combo_p + C.L5P_COMBO_PER_KM * km, C.L5P_COMBO_MAX)

    def distance_for_level(self, i: int) -> float:
        """Distance where level i starts (endless); 0 for fixed levels."""
        if not self.multi or i <= 0:
            return 0.0
        return self.starts[i]


# ------------------------------------------------------------------------------------
# Hazards (immutable once created)
# ------------------------------------------------------------------------------------
class Static:
    """Building, storm dip, downdraft zone, or (v2.0) a tree group, street lights, power lines, a
    kite or a crane built from ``parts``. Fixed in world x."""

    __slots__ = ("uid", "kind", "x0", "x1", "y0", "y1", "xc", "r", "parts")

    def __init__(self, uid, kind, x0, x1, y0, y1, xc=0.0, r=0.0, parts=()):
        self.uid = uid
        self.kind = kind
        self.x0 = x0          # left edge (of the blocking parts)
        self.x1 = x1          # right edge
        self.y0 = y0          # bottom (storm: lowest point of the dip; parts: lowest blocked point)
        self.y1 = y1          # top (mountain: peak; storm/downdraft: ceiling; parts: highest point)
        self.xc = xc          # mountain peak x; parts: where the obstacle is tightest
        self.r = r            # storm ramp length
        self.parts = parts    # PART_KINDS: (BOX, x0, y0, x1, y1) and (WIRE, ax, ay, bx, by, half thickness)

    def key(self):
        return (self.uid, self.kind, round(self.x0, 9), round(self.x1, 9), round(self.y0, 9),
                round(self.y1, 9))


class Mover:
    """Bird or traffic. Dormant until activated (``xr`` is None).

    Height at time t (SPEC v2.3): ``base + amp * sin(omega t + phase + fm * sin(omega_m t + phase_m))
    + dive * max(0, sin(omega_d t + phase_d)) ** 16``. ``fm`` makes the rhythm wander (the swing runs
    ahead of and behind a steady sine by up to fm radians); ``dive`` adds a short sudden drop
    (dive < 0) or rise (dive > 0) once per 2 pi / omega_d seconds. fm = dive = 0 is the plain sine
    path of SPEC v1.x-v2.2; planes have amp = 0 (level flight).
    """

    __slots__ = ("uid", "kind", "sub", "w", "h", "base", "amp", "omega", "phase", "s", "xe", "xr", "tr",
                 "fm", "omega_m", "phase_m", "dive", "omega_d", "phase_d")

    def __init__(self, uid, kind, sub, w, h, base, amp, omega, phase, s, xe, xr=None, tr=0.0,
                 fm=0.0, omega_m=0.0, phase_m=0.0, dive=0.0, omega_d=0.0, phase_d=0.0):
        self.uid = uid
        self.kind = kind
        self.sub = sub        # traffic: 0 = drone, 1 = balloon; bird: 0
        self.w = w
        self.h = h
        self.base = base      # centre altitude of the path
        self.amp = amp        # sine amplitude (birds; v2.1 traffic drift), 0 for planes
        self.omega = omega    # 2*pi/period
        self.phase = phase
        self.s = s            # own speed toward the drone (world x decreases by s per second)
        self.xe = xe          # left edge where the drone should meet it
        self.xr = xr          # left edge at time tr (None while dormant)
        self.tr = tr
        self.fm = fm          # v2.3: rhythm wander (radians), its angular frequency and phase
        self.omega_m = omega_m
        self.phase_m = phase_m
        self.dive = dive      # v2.3: dive depth (m; < 0 down, > 0 up), its angular frequency and phase
        self.omega_d = omega_d
        self.phase_d = phase_d

    def copy(self, **changes) -> "Mover":
        m = Mover.__new__(Mover)
        for k in Mover.__slots__:
            setattr(m, k, getattr(self, k))
        for k, v in changes.items():
            setattr(m, k, v)
        return m

    def activated(self, nose_x: float, v_now: float, t_now: float) -> "Mover":
        # Place it so that, at the current speed, the drone's nose meets its left edge at xe.
        xr = self.xe + self.s * (self.xe - nose_x) / v_now
        return self.copy(xr=xr, tr=t_now)

    def x0_at(self, t: float) -> float:
        return self.xr - self.s * (t - self.tr)

    @property
    def still(self) -> bool:
        """True if it never moves up or down (planes)."""
        return self.amp == 0.0 and self.dive == 0.0

    @property
    def down(self) -> float:
        """How far below ``base`` its centre can go."""
        return self.amp + (-self.dive if self.dive < 0.0 else 0.0)

    @property
    def up(self) -> float:
        """How far above ``base`` its centre can go."""
        return self.amp + (self.dive if self.dive > 0.0 else 0.0)

    def y_at(self, t: float) -> float:
        if self.amp == 0.0 and self.dive == 0.0:
            return self.base
        a = self.omega * t + self.phase
        if self.fm != 0.0:
            a += self.fm * math.sin(self.omega_m * t + self.phase_m)
        y = self.base + self.amp * math.sin(a)
        if self.dive != 0.0:
            q = math.sin(self.omega_d * t + self.phase_d)
            if q > 0.0:
                q *= q
                q *= q
                q *= q
                y += self.dive * q * q
        return y

    def vy_at(self, t: float) -> float:
        if self.amp == 0.0 and self.dive == 0.0:
            return 0.0
        a = self.omega * t + self.phase
        da = self.omega
        if self.fm != 0.0:
            a += self.fm * math.sin(self.omega_m * t + self.phase_m)
            da += self.fm * self.omega_m * math.cos(self.omega_m * t + self.phase_m)
        v = self.amp * da * math.cos(a)
        if self.dive != 0.0:
            b = self.omega_d * t + self.phase_d
            q = math.sin(b)
            if q > 0.0:
                q2 = q * q
                q4 = q2 * q2
                q8 = q4 * q4
                v += self.dive * 16.0 * q8 * q4 * q2 * q * self.omega_d * math.cos(b)
        return v

    def key(self):
        return (self.uid, self.kind, self.sub, round(self.xe, 9),
                None if self.xr is None else round(self.xr, 9), round(self.tr, 9), round(self.base, 9))


# ------------------------------------------------------------------------------------
# Shape helpers shared by collision, sensors and the passability check
# ------------------------------------------------------------------------------------
def tent_max(h: Static, a: float, b: float) -> float:
    """Highest point of a mountain over the x-interval [a, b] (0 if no overlap)."""
    if b < h.x0 or a > h.x1:
        return 0.0
    if a < h.x0:
        a = h.x0
    if b > h.x1:
        b = h.x1
    if a <= h.xc <= b:
        return h.y1
    if b < h.xc:
        return h.y1 * (b - h.x0) / (h.xc - h.x0)
    return h.y1 * (h.x1 - a) / (h.x1 - h.xc)


def storm_ceiling_at(h: Static, x: float) -> float:
    if x <= h.x0 or x >= h.x1:
        return C.CEILING
    depth = C.CEILING - h.y0
    if x < h.x0 + h.r:
        return C.CEILING - depth * (x - h.x0) / h.r
    if x > h.x1 - h.r:
        return C.CEILING - depth * (h.x1 - x) / h.r
    return h.y0


def storm_min(h: Static, a: float, b: float) -> float:
    """Lowest storm ceiling over [a, b] (CEILING if no overlap)."""
    if b <= h.x0 or a >= h.x1:
        return C.CEILING
    if a < h.x0:
        a = h.x0
    if b > h.x1:
        b = h.x1
    if b >= h.x0 + h.r and a <= h.x1 - h.r:
        return h.y0
    if b < h.x0 + h.r:
        return storm_ceiling_at(h, b)
    return storm_ceiling_at(h, a)


BOX, WIRE = 0, 1     # Static.parts: x at indices 1 and 3 in both


def parts_band(h: Static, a: float, b: float):
    """Height interval (lo, hi) that a parts-built static blocks over the x-interval [a, b] (the
    hull of its parts there), or None. Each v2.0 obstacle's parts touch, so the hull is exact for
    passability: the space between them cannot be reached (a trunk under a canopy, a cable under a
    jib, a string under a kite)."""
    lo, hi = math.inf, -math.inf
    for p in h.parts:
        if p[0] == BOX:
            if p[3] < a or p[1] > b:
                continue
            if p[2] < lo:
                lo = p[2]
            if p[4] > hi:
                hi = p[4]
            continue
        _, ax, ay, bx, by, r = p
        if ax > bx:
            ax, ay, bx, by = bx, by, ax, ay
        if bx < a - r or ax > b + r:
            continue
        if bx - ax < 1e-9:
            ya, yb = ay, by
        else:
            xa = min(max(a, ax), bx)
            xb = max(min(b, bx), ax)
            k = (by - ay) / (bx - ax)
            ya, yb = ay + k * (xa - ax), ay + k * (xb - ax)
        if ya > yb:
            ya, yb = yb, ya
        if ya - r < lo:
            lo = ya - r
        if yb + r > hi:
            hi = yb + r
    return (lo, hi) if lo <= hi else None


def parts_boxes_wires(h: Static):
    """(boxes [(x0, y0, x1, y1)], wires [(ax, ay, bx, by)]) of a parts-built static."""
    boxes = [p[1:5] for p in h.parts if p[0] == BOX]
    wires = [p[1:5] for p in h.parts if p[0] == WIRE]
    return boxes, wires


def mountain_segments(h: Static):
    return ((h.x0, 0.0, h.xc, h.y1), (h.xc, h.y1, h.x1, 0.0))


def storm_segments(h: Static):
    return ((h.x0, C.CEILING, h.x0 + h.r, h.y0), (h.x0 + h.r, h.y0, h.x1 - h.r, h.y0),
            (h.x1 - h.r, h.y0, h.x1, C.CEILING))


# ------------------------------------------------------------------------------------
# Passability footprints
# ------------------------------------------------------------------------------------
FOOT_PAD = 0.5 * C.CHECK_DX   # SPEC v2.3: statics block half a sweep step wider, so a narrow spot
                              # between two sweep samples (a tree's top under a storm ramp) is seen


class Footprint:
    """What one hazard blocks for the drone *centre* at a given x (hitbox width included; statics
    widened by FOOT_PAD on each side)."""

    __slots__ = ("kind", "fx0", "fx1", "h", "lo", "hi")

    def __init__(self, kind, fx0, fx1, h=None, lo=0.0, hi=0.0):
        self.kind = kind
        self.fx0 = fx0
        self.fx1 = fx1
        self.h = h            # Static for mountain/storm
        self.lo = lo
        self.hi = hi

    def band(self, x: float):
        k = self.kind
        w = C.HALF_W + FOOT_PAD
        if k in C.PART_KINDS:
            return parts_band(self.h, x - w, x + w)
        if k == C.MOUNTAIN:
            top = tent_max(self.h, x - w, x + w)
            return (0.0, top) if top > 0.0 else None
        if k == C.STORM:
            low = storm_min(self.h, x - w, x + w)
            return (low, C.CEILING) if low < C.CEILING else None
        if k == C.DOWNDRAFT:
            return None
        return (self.lo, self.hi)


def footprint_of(h) -> Footprint:
    if isinstance(h, Static):
        w = C.HALF_W + FOOT_PAD
        if h.kind == C.BUILDING:
            return Footprint(C.BUILDING, h.x0 - w, h.x1 + w, lo=0.0, hi=h.y1)
        return Footprint(h.kind, h.x0 - w, h.x1 + w, h=h)
    # Mover: conservative band over the whole path (swing and dive), wider x span for arrival-time slack
    # (planes are fast: their slack grows with their own speed).
    m = C.MOVER_X_MARGIN + C.HALF_W
    if h.kind == C.PLANE:
        m += C.PLANE_X_SLACK_S * abs(h.s)
    return Footprint(h.kind, h.xe - m, h.xe + h.w + m, lo=h.base - h.down - h.h / 2, hi=h.base + h.up + h.h / 2)


def free_centre(x: float, fps) -> list:
    """Free intervals for the drone centre at x: gaps >= CORRIDOR_MIN, shrunk by the hitbox."""
    bands = []
    for fp in fps:
        if fp.fx0 <= x <= fp.fx1:
            b = fp.band(x)
            if b is not None:
                bands.append(b)
    bands.sort()
    gaps = []
    cur = 0.0
    for lo, hi in bands:
        lo -= C.EDGE_MARGIN
        hi += C.EDGE_MARGIN
        if lo > cur:
            gaps.append((cur, lo))
        if hi > cur:
            cur = hi
    if cur < C.CEILING:
        gaps.append((cur, C.CEILING))
    shrink = C.HALF_H + C.EDGE_MARGIN
    return [(g0 + shrink, g1 - shrink) for g0, g1 in gaps if g1 - g0 >= C.CORRIDOR_MIN]


def in_downdraft(x: float, fps) -> bool:
    for fp in fps:
        if fp.kind == C.DOWNDRAFT and fp.fx0 <= x <= fp.fx1:
            return True
    return False


def _expand(R, dn: float, up: float):
    out = []
    for lo, hi in R:
        lo -= dn
        hi += up
        if out and lo <= out[-1][1]:
            if hi > out[-1][1]:
                out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def _intersect(A, B):
    out = []
    i = j = 0
    while i < len(A) and j < len(B):
        lo = A[i][0] if A[i][0] > B[j][0] else B[j][0]
        hi = A[i][1] if A[i][1] < B[j][1] else B[j][1]
        if lo <= hi:
            out.append((lo, hi))
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return out


def viable_back(k0: int, k1: int, fps, schedule: Schedule):
    """Backward viability: altitudes at grid k0 from which the drone can fly to grid k1."""
    dx = C.CHECK_DX
    V = free_centre(k1 * dx, fps)
    for k in range(k1, k0, -1):
        x = (k - 1) * dx
        tau = dx / schedule.speed_at(x - C.DRONE_X)
        up = (C.DOWNDRAFT_CLIMB_RATE if in_downdraft(x, fps) else C.CLIMB_RATE) * tau
        # y at x is viable if some y' in V(x + dx) lies in [y - descent, y + climb]
        V = _intersect(_expand(V, up, C.CLIMB_RATE * tau), free_centre(x, fps))
        if not V:
            return V
    return V


def mid_band(F, half: float):
    """Middle +/- half of every free gap: where a sensible pilot passes an obstacle."""
    out = []
    for lo, hi in F:
        m = 0.5 * (lo + hi)
        out.append((max(lo, m - half), min(hi, m + half)))
    return out


def tight_x(statics, movers):
    """x where an obstacle constrains the corridor most (None for downdrafts)."""
    xs = []
    for h in statics:
        if h.kind == C.MOUNTAIN or h.parts:
            xs.append(h.xc)
        elif h.kind != C.DOWNDRAFT:
            xs.append(0.5 * (h.x0 + h.x1))
    for m in movers:
        xs.append(m.xe + 0.5 * m.w)
    return sum(xs) / len(xs) if xs else None


def covers(V, A) -> bool:
    """True if the interval set A is contained in the interval set V."""
    for lo, hi in A:
        if not any(v0 <= lo + 1e-9 and hi - 1e-9 <= v1 for v0, v1 in V):
            return False
    return True


def visible_x(m: Mover, v: float) -> float:
    """Drone-centre x where a dormant mover first comes into sensor range (nominal speed v):
    its left edge is SENSOR_RANGE ahead of the drone's nose and closing at v + s."""
    closing = v + m.s
    if closing <= 1e-6:
        return -math.inf
    return m.xe - C.HALF_W - C.SENSOR_RANGE * v / closing


# reflexes tried by the plane escape check: (first action, wait frames, follow-up action)
_ESCAPES = tuple([(C.HOLD, 0, C.HOLD), (C.ASCEND, 0, C.ASCEND), (C.DESCEND, 0, C.DESCEND)] +
                 [(a, w, b) for w in C.ESCAPE_WAITS for a, b in
                  ((C.JUMP, C.HOLD), (C.JUMP, C.ASCEND), (C.DUCK, C.HOLD), (C.DUCK, C.DESCEND))])


def _escape_flight(y, vy, plan, frees, dds, viable_end) -> bool:
    """Fly one reflex with the simulator's own vertical dynamics (no wind but downdrafts).
    frees[j] / dds[j]: free drone-centre intervals / downdraft at the x of frame j."""
    first, wait, follow = plan
    burst, burst_v = 0, 0.0
    # while reacting, the drone keeps doing what it was doing (climbing, sinking or holding)
    vt_keep = C.V_ASCEND if vy > 0.5 else (C.V_DESCEND if vy < -0.5 else C.V_HOLD)
    react = C.REACTION_FRAMES
    for j in range(len(frees)):
        if j < react + wait:
            vt = vt_keep if j < react else C.V_HOLD
        elif burst > 0:
            vt = burst_v
        else:
            a = first if j == react + wait else (follow if first in (C.JUMP, C.DUCK) else first)
            if first in (C.JUMP, C.DUCK) and j == react + wait:
                burst = C.JUMP_BURST if first == C.JUMP else C.DUCK_BURST
                burst_v = C.JUMP_V if first == C.JUMP else C.DUCK_V
                vy = burst_v
                vt = burst_v
            elif a == C.ASCEND:
                vt = C.V_ASCEND
            elif a == C.DESCEND:
                vt = C.V_DESCEND
            else:
                vt = C.V_HOLD
        acc = C.CTRL_GAIN * (vt - vy)
        acc = C.A_MAX if acc > C.A_MAX else (-C.A_MAX if acc < -C.A_MAX else acc)
        vy += (acc + (C.DOWNDRAFT_ACCEL if dds[j] else 0.0)) * C.DT
        y += vy * C.DT
        if burst > 0:
            burst -= 1
        if not any(lo <= y <= hi for lo, hi in frees[j]):
            return False
    return any(lo - 1e-9 <= y <= hi + 1e-9 for lo, hi in viable_end)


def plane_escape_ok(P, x_start: float, x_stop: float, fps, schedule: Schedule, k_end: int) -> bool:
    """True if, from every sampled altitude of P (and every sampled vertical speed) at drone-centre
    x_start, one reflex in _ESCAPES flies past x_stop with real physics and ends where the rest of
    the known world is still passable (rate-model viability)."""
    v = schedule.speed_at(x_start - C.DRONE_X)
    step = v * C.DT
    n = max(1, int(math.ceil((x_stop - x_start) / step)))
    xs = [x_start + step * (j + 1) for j in range(n)]
    frees = [free_centre(x, fps) for x in xs]
    dds = [in_downdraft(x, fps) for x in xs]
    k_stop = int(math.ceil(xs[-1] / C.CHECK_DX))
    V_end = viable_back(k_stop, max(k_end, k_stop), fps, schedule) if k_stop < k_end else free_centre(k_stop * C.CHECK_DX, fps)
    # the last flown x lies between grid points: a 0.5 m step is under 0.04 s, so test against
    # the viable set at the next grid point widened by what 3 m/s covers in that time
    pad = C.CLIMB_RATE * (k_stop * C.CHECK_DX - xs[-1]) / v
    V_end = [(lo - pad, hi + pad) for lo, hi in V_end]
    samples = []
    for lo, hi in P:
        samples.extend((lo, 0.5 * (lo + hi), hi) if hi - lo > 1e-6 else (lo,))
    for y0 in samples:
        for vy0 in C.ESCAPE_VY:
            if not any(_escape_flight(y0, vy0, plan, frees, dds, V_end) for plan in _ESCAPES):
                return False
    return True


def sweep(R, k0: int, k1: int, fps, schedule: Schedule, record_at: int | None = None):
    """Advance the reachable set from grid index k0 to k1. Returns (ok, R_end, R_at_record)."""
    dx = C.CHECK_DX
    rec = R if record_at == k0 else None
    for k in range(k0, k1):
        x = k * dx
        tau = dx / schedule.speed_at(x - C.DRONE_X)
        up = (C.DOWNDRAFT_CLIMB_RATE if in_downdraft(x, fps) else C.CLIMB_RATE) * tau
        R = _intersect(_expand(R, C.CLIMB_RATE * tau, up), free_centre(x + dx, fps))
        if not R:
            return False, R, rec
        if record_at == k + 1:
            rec = R
    return True, R, rec


# ------------------------------------------------------------------------------------
# Layout generator
# ------------------------------------------------------------------------------------
class RNGRef:
    """Copy-on-write holder for a numpy Generator, so clone() never copies RNG state eagerly."""

    __slots__ = ("rng", "shared")

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.shared = False

    def own(self) -> "RNGRef":
        if self.shared:
            import copy
            return RNGRef(copy.deepcopy(self.rng))
        return self


class LayoutGenerator:
    def __init__(self, schedule: Schedule, rng: np.random.Generator, record: bool = False):
        self.schedule = schedule
        self.rng = RNGRef(rng)
        self.next_start = C.DRONE_X + C.START_CLEAR_S * schedule.speed_at(0.0)
        self.commit_k = int(round(C.DRONE_X / C.CHECK_DX))
        self.R = [(C.START_Y, C.START_Y)]
        self.tight_k = self.commit_k   # grid index of the previous obstacle's tightest point
        self.P = [(C.START_Y - C.ROBUST_HALF, C.START_Y + C.ROBUST_HALF)]   # plausible altitudes there
        self.open = []                 # footprints that reach past the commit point
        self.uid = 0
        self.placed = 0
        self.fallbacks = 0
        self.tries = 0
        self.mover_rejects = 0          # candidates refused by the first-sight check
        self.plane_rejects = 0          # candidates refused by the plane escape/spacing checks
        self.last_plane_xv = -math.inf  # drone x where the last accepted plane came into view
        self._pending_checks = ()
        self.combos = 0
        self.combos_tried = 0
        self.record = record
        self.log = [] if record else None   # every accepted hazard (dormant movers as created)
        self.starts = [] if record else None
        self.plane_checks = [] if record else None   # (plane, first-sight x, P band) per accepted plane

    def clone(self) -> "LayoutGenerator":
        g = LayoutGenerator.__new__(LayoutGenerator)
        d = self.__dict__.copy()
        self.rng.shared = True
        d["open"] = list(self.open)
        if self.record:
            d["log"] = list(self.log)
            d["starts"] = list(self.starts)
            d["plane_checks"] = list(self.plane_checks)
        g.__dict__ = d
        return g

    # -- sampling ---------------------------------------------------------------------
    def _new_uid(self) -> int:
        self.uid += 1
        return self.uid

    def _sample(self, kind: int, xs: float, rng: np.random.Generator):
        u = rng.uniform
        hz = C.HZ
        statics, movers = [], []
        if kind == C.BUILDING:
            w = u(*hz["building"]["width"])
            h = u(*hz["building"]["height"])
            statics.append(Static(0, C.BUILDING, xs, xs + w, 0.0, h))
        elif kind == C.MOUNTAIN:
            base = u(*hz["mountain"]["base"])
            peak = u(*hz["mountain"]["peak"])
            statics.append(Static(0, C.MOUNTAIN, xs, xs + base, 0.0, peak, xc=xs + base / 2))
        elif kind == C.STORM:
            w = u(*hz["storm"]["width"])
            low = u(*hz["storm"]["lowest"])
            statics.append(Static(0, C.STORM, xs, xs + w, low, C.CEILING, r=w * C.STORM_RAMP_FRAC))
        elif kind == C.DOWNDRAFT:
            w = u(*hz["downdraft"]["width"])
            statics.append(Static(0, C.DOWNDRAFT, xs, xs + w, 0.0, C.CEILING))
        elif kind == C.BIRD:
            b = hz["bird"]
            n = int(rng.integers(b["flock"][0], b["flock"][1] + 1))
            base0 = u(*b["altitude"])
            amp0 = u(*b["amplitude"])
            omega = 2 * math.pi / u(*b["period_s"])
            s = u(*b["own_speed"])
            bw, bh = b["size"]
            x = xs
            for i in range(n):
                if i:
                    x += u(*b["spacing_x"])
                base = min(max(base0 + u(-b["spread_y"], b["spread_y"]), b["altitude"][0]), b["altitude"][1])
                room = C.CEILING - 0.1 - bh / 2 - base             # stay under the sky ceiling
                amp = min(amp0, room)
                # SPEC v2.3: each bird's rhythm wanders, and some make a short dive (or pop up)
                fm, om = u(*b["rhythm_change"]), 2 * math.pi / u(*b["rhythm_period_s"])
                dive, od = 0.0, 0.0
                if rng.random() < b["dive_prob"]:
                    dive = u(*b["dive_depth"])
                    if rng.random() < b["dive_up_prob"]:
                        dive = min(dive, max(room - amp, 0.0))
                    else:
                        dive = -dive
                    od = 2 * math.pi / u(*b["dive_every_s"])
                movers.append(Mover(0, C.BIRD, 0, bw, bh, base, amp, omega, u(0, 2 * math.pi), s, x,
                                    fm=fm, omega_m=om, phase_m=u(0, 2 * math.pi),
                                    dive=dive, omega_d=od, phase_d=u(0, 2 * math.pi)))
        elif kind == C.TRAFFIC:
            t = hz["traffic"]
            balloon = rng.random() < t["balloon_prob"]
            w, h = t["balloon_size"] if balloon else t["drone_size"]
            alt = u(*t["altitude"])
            alt = min(alt, C.CEILING - 0.1 - h / 2)
            s = u(*t["own_speed"])
            amp = min(u(*t["drift_amplitude"]), C.CEILING - 0.1 - h / 2 - alt)   # v2.1: drifts, under the ceiling
            omega = 2 * math.pi / u(*t["drift_period_s"])
            fm, om = u(*t["rhythm_change"]), 2 * math.pi / u(*t["rhythm_period_s"])   # v2.3: wandering rhythm
            movers.append(Mover(0, C.TRAFFIC, int(balloon), w, h, alt, amp, omega, u(0, 2 * math.pi), s, xs,
                                fm=fm, omega_m=om, phase_m=u(0, 2 * math.pi)))
        elif kind in C.PART_KINDS:
            statics.append(self._sample_parts(kind, xs, rng))
        elif kind == C.PLANE:
            p = hz["plane"]
            prop = rng.random() < p["prop_prob"]
            (w_lo, w_hi), h = p["prop_size"] if prop else p["jet_size"]
            w = u(w_lo, w_hi)
            alt = min(u(*p["altitude"]), C.CEILING - 0.1 - h / 2)
            s = u(*p["own_speed"])
            movers.append(Mover(0, C.PLANE, int(prop), w, h, alt, 0.0, 0.0, 0.0, s, xs))
        else:
            raise ValueError(kind)
        return statics, movers

    def _sample_parts(self, kind: int, xs: float, rng) -> Static:
        """A v2.0 obstacle (SPEC §3, game.yaml) with its leftmost part at xs."""
        u = rng.uniform
        hz = C.HZ
        parts = []
        if kind == C.TREE:
            t = hz["tree"]
            n = int(rng.integers(t["count"][0], t["count"][1] + 1))
            x, top, xc = xs, -1.0, xs
            for _ in range(n):
                H = u(*t["height"])
                cw = u(*t["canopy_width"])
                cb = H * u(*t["canopy_bottom"])
                tw = u(*t["trunk_width"])
                cx = x + cw / 2
                ch = H - cb
                parts.append((BOX, cx - tw / 2, 0.0, cx + tw / 2, cb))     # trunk
                # a rounded canopy: three stacked boxes, narrower at the bottom and the top
                parts.append((BOX, x + 0.12 * cw, cb, x + 0.88 * cw, cb + 0.22 * ch))
                parts.append((BOX, x, cb + 0.22 * ch, x + cw, cb + 0.75 * ch))
                parts.append((BOX, x + 0.15 * cw, cb + 0.75 * ch, x + 0.85 * cw, H))
                if H > top:
                    top, xc = H, cx
                x += cw * u(*t["step"])
            y0, y1 = 0.0, top
        elif kind == C.POLE:
            q = hz["pole"]
            n = int(rng.integers(q["count"][0], q["count"][1] + 1))
            H = u(*q["height"])
            w, arm, at = q["width"], q["arm"], q["arm_thickness"]
            lw, lh = q["lamp"]
            side = -1.0 if rng.random() < 0.5 else 1.0      # lamp arm toward the drone or away
            xp = xs
            for i in range(n):
                if i:
                    xp += u(*q["spacing"])
                end = xp + side * arm
                parts.append((BOX, xp - w / 2, 0.0, xp + w / 2, H))                     # pole
                parts.append((BOX, min(xp, end), H - at, max(xp, end), H))              # arm
                parts.append((BOX, end - lw / 2, H - at - lh, end + lw / 2, H - at))    # lamp head
            y0, y1, xc = 0.0, H, 0.0
        elif kind == C.POWERLINE:
            q = hz["powerline"]
            L, A, S = u(*q["span"]), u(*q["height"]), u(*q["sag"])
            r = q["wire_r"]
            for top in (A, A - q["wire_gap"]):
                pts = [(xs + L * f, top - S * (1.0 - (2.0 * f - 1.0) ** 2)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
                for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                    parts.append((WIRE, ax, ay, bx, by, r))
            y0, y1, xc = A - q["wire_gap"] - S - r, A + r, xs + L / 2
        elif kind == C.KITE:
            q = hz["kite"]
            kc = u(*q["height"])
            kw, kh = q["size"]
            d = u(*q["anchor_dx"]) * (1.0 if rng.random() < 0.5 else -1.0)   # anchor ahead (+) or behind (-)
            xk, r = xs, q["string_r"]
            # the kite's outline (a diamond, cross point 60% up) and its tail are wires too, so the
            # drawn kite is exactly what the drone can hit
            top, right, bot, left = (xk, kc + kh / 2), (xk + kw / 2, kc + 0.1 * kh), (xk, kc - kh / 2), (xk - kw / 2, kc + 0.1 * kh)
            for (ax, ay), (bx, by) in ((top, right), (right, bot), (bot, left), (left, top)):
                parts.append((WIRE, ax, ay, bx, by, r))
            tail_end = (xk - 0.5 * d / abs(d) * 0.12, kc - kh / 2 - q["tail"])
            parts.append((WIRE, bot[0], bot[1], tail_end[0], tail_end[1], r))                 # tail
            parts.append((WIRE, xk, kc - kh / 2, xk + d, 0.0, r))                             # string
            y0, y1, xc = 0.0, kc + kh / 2 + r, xk
        elif kind == C.CRANE:
            q = hz["crane"]
            jb = u(*q["jib_height"])
            L = u(*q["jib_length"])
            hb = u(*q["hook_bottom"])
            hw, hh = q["hook_size"]
            xh = xs + u(*q["hook_at"]) * L
            parts.append((BOX, xs, jb, xs + L, jb + q["jib_thickness"]))                     # jib
            parts.append((WIRE, xh, hb + hh, xh, jb, q["cable_r"]))                          # hook cable
            parts.append((BOX, xh - hw / 2, hb, xh + hw / 2, hb + hh))                       # hook block
            y0, y1, xc = hb, jb + q["jib_thickness"], xh
        else:
            raise ValueError(kind)
        x0 = min(min(p[1], p[3]) - (p[5] if p[0] == WIRE else 0.0) for p in parts)
        x1 = max(max(p[1], p[3]) + (p[5] if p[0] == WIRE else 0.0) for p in parts)
        dx = xs - x0                               # leftmost blocking part at xs
        parts = tuple((p[0], p[1] + dx, p[2], p[3] + dx, p[4]) + p[5:] for p in parts)
        if kind == C.POLE:
            xc = 0.5 * (xs + x1 + dx)
        else:
            xc += dx
        return Static(0, kind, xs, x1 + dx, y0, y1, xc=xc, parts=parts)

    # Combo partners: what can share the same stretch of sky with a primary obstacle (v2.1: planes
    # back, high above buildings and trees, never in a storm; no cranes or power lines, which keep
    # entries only as primaries with weight 0 in levels.yaml, like mountains).
    _PARTNERS = {
        C.BUILDING: ((C.BIRD, C.TRAFFIC, C.PLANE, C.STORM, C.DOWNDRAFT), (0.38, 0.33, 0.07, 0.16, 0.06)),
        C.MOUNTAIN: ((C.BIRD, C.TRAFFIC, C.STORM, C.DOWNDRAFT), (0.40, 0.36, 0.17, 0.07)),
        C.TREE: ((C.BIRD, C.TRAFFIC, C.PLANE, C.STORM, C.DOWNDRAFT), (0.38, 0.33, 0.07, 0.16, 0.06)),
        C.POLE: ((C.BIRD, C.TRAFFIC, C.STORM), (0.44, 0.37, 0.19)),
        C.POWERLINE: ((C.TREE, C.POLE, C.BUILDING, C.BIRD, C.TRAFFIC), (0.25, 0.20, 0.20, 0.20, 0.15)),
        C.CRANE: ((C.BUILDING, C.BIRD, C.TRAFFIC), (0.50, 0.25, 0.25)),
        C.KITE: ((C.TREE, C.BIRD, C.TRAFFIC), (0.30, 0.35, 0.35)),
        C.STORM: ((C.BUILDING, C.TREE, C.POLE, C.BIRD, C.TRAFFIC), (0.30, 0.25, 0.10, 0.20, 0.15)),
        C.BIRD: ((C.BUILDING, C.TREE, C.POLE, C.STORM), (0.39, 0.28, 0.11, 0.22)),
        C.TRAFFIC: ((C.BUILDING, C.TREE, C.POLE, C.STORM), (0.39, 0.28, 0.11, 0.22)),
        C.PLANE: ((C.BUILDING, C.TREE), (0.55, 0.45)),
        C.DOWNDRAFT: ((C.BUILDING, C.TREE, C.STORM, C.BIRD, C.TRAFFIC, C.POLE, C.PLANE),
                      (0.24, 0.19, 0.15, 0.19, 0.14, 0.05, 0.04)),
    }

    @staticmethod
    def _span(statics, movers):
        lo, hi = math.inf, -math.inf
        for h in statics:
            lo, hi = min(lo, h.x0), max(hi, h.x1)
        for m in movers:
            lo, hi = min(lo, m.xe), max(hi, m.xe + m.w)
        return lo, hi

    @staticmethod
    def _shift(statics, movers, dx: float) -> None:
        for h in statics:           # fresh objects, not placed yet: safe to move
            h.x0 += dx
            h.x1 += dx
            if h.kind == C.MOUNTAIN or h.parts:
                h.xc += dx
            if h.parts:
                h.parts = tuple((p[0], p[1] + dx, p[2], p[3] + dx, p[4]) + p[5:] for p in h.parts)
        for m in movers:
            m.xe += dx

    def _combo(self, kind: int, xs: float, statics, movers, rng):
        """Add a partner hazard centred on the primary; the combo starts at xs."""
        kinds, weights = self._PARTNERS[kind]
        r, acc, k2 = rng.random(), 0.0, kinds[-1]
        for k, w in zip(kinds, weights):
            acc += w
            if r <= acc:
                k2 = k
                break
        s2, m2 = self._sample(k2, xs, rng)
        lo1, hi1 = self._span(statics, movers)
        lo2, hi2 = self._span(s2, m2)
        jitter = rng.uniform(-0.3, 0.3) * (hi1 - lo1)
        self._shift(s2, m2, 0.5 * (lo1 + hi1) - 0.5 * (lo2 + hi2) + jitter)
        statics, movers = statics + s2, movers + m2
        lo, _ = self._span(statics, movers)
        self._shift(statics, movers, xs - lo)
        return statics, movers

    def _pick_kind(self, spec: LevelSpec, rng) -> int:
        r = rng.random()
        for k, c in zip(spec.kinds, spec.cum):
            if r <= c:
                return k
        return spec.kinds[-1]

    def _movers_fair(self, movers, fps, k_end: int, v: float, sched) -> bool:
        """SPEC v1.7 reflex checks for a candidate with moving hazards (see game.yaml)."""
        dx = C.CHECK_DX
        planes = [m for m in movers if m.kind == C.PLANE]
        xv = min(visible_x(m, v) for m in movers)       # first sight of this candidate's movers
        for m in planes:
            if visible_x(m, v) - self.last_plane_xv < v * C.PLANE_SPACING_S:
                self.plane_rejects += 1
                return False
        kv = int(math.floor(xv / dx))
        if kv <= self.tight_k:
            Pv = _intersect(self.P, free_centre(self.tight_k * dx, fps)) if planes else None
            kv = self.tight_k
        else:
            # the pilot holds its plausible band until the movers come into view
            Pv = _intersect(self.P, free_centre(kv * dx, fps))
            if Pv and not covers(viable_back(kv, k_end, fps, sched), Pv):
                self.mover_rejects += 1
                return False
        if planes and Pv:
            x_stop = max(footprint_of(m).fx1 for m in planes)
            if not plane_escape_ok(Pv, kv * dx, x_stop, fps, sched, k_end):
                self.plane_rejects += 1
                return False
            self._pending_checks = [(m, kv * dx, x_stop, Pv) for m in planes]
        return True

    # -- main loop --------------------------------------------------------------------
    def generate_until(self, x_target: float, out_statics: list, out_dormant: list) -> None:
        while self.next_start < x_target:
            self._place(self.next_start, out_statics, out_dormant)

    def _place(self, xs: float, out_statics: list, out_dormant: list) -> None:
        self.rng = self.rng.own()
        rng = self.rng.rng
        sched = self.schedule
        d = xs - C.DRONE_X
        spec = sched.spec_at(d)
        v = sched.speed_at(d)
        gap_lo, gap_hi, combo_p = sched.density_at(d, spec)
        commit_k = max(int(math.floor(xs / C.CHECK_DX)), self.commit_k)
        accepted = None
        for attempt in range(C.MAX_TRIES):
            self.tries += 1
            kind = self._pick_kind(spec, rng)
            statics, movers = self._sample(kind, xs, rng)
            # combos only in the first half of the tries, so a hard spot falls back to singles
            if combo_p > 0.0 and attempt < C.MAX_TRIES // 2 and rng.random() < combo_p:
                statics, movers = self._combo(kind, xs, statics, movers, rng)
                self.combos_tried += 1
            new_fps = [footprint_of(h) for h in statics] + [footprint_of(m) for m in movers]
            fps = self.open + new_fps
            end_x = max(fp.fx1 for fp in fps)
            k_end = max(int(math.ceil(end_x / C.CHECK_DX)) + 1, commit_k)
            ok, _, R_commit = sweep(self.R, self.commit_k, k_end, fps, sched, record_at=commit_k)
            if ok and C.ROBUST:
                # Robust "consecutive corridors": a reflex pilot passes each obstacle near the
                # middle of its corridor. From anywhere within +/- ROBUST_HALF of that middle at
                # the previous obstacle's tightest point, the new obstacle must still be passable.
                # (The forward sweep alone assumes perfect foresight.)
                kt = self.tight_k
                P = _intersect(self.P, free_centre(kt * C.CHECK_DX, fps))
                ok = bool(P) and covers(viable_back(kt, k_end, fps, sched), P)
                if ok and movers:
                    ok = self._movers_fair(movers, fps, k_end, v, sched)
            if ok:
                accepted = (statics, movers, fps, R_commit)
                break
        if accepted is None:
            # Nothing fits here: leave an empty stretch (always passable) and try again later.
            self.fallbacks += 1
            self.next_start = xs + v * C.MIN_GAP_S * 0.5
            return
        statics, movers, fps, R_commit = accepted
        for m in movers:
            if m.kind == C.PLANE:
                self.last_plane_xv = max(self.last_plane_xv, visible_x(m, v))
        if self.record and movers:
            self.plane_checks.extend(getattr(self, "_pending_checks", ()))
        self._pending_checks = ()
        for h in statics:
            h.uid = self._new_uid()
            out_statics.append(h)
        for m in movers:
            m.uid = self._new_uid()
            out_dormant.append(m)
        if self.record:
            self.log.extend(statics)
            self.log.extend(movers)
            self.starts.append((xs, v))
        self.placed += 1
        if len(statics) + len({m.xe for m in movers if m.kind == C.TRAFFIC}) + (1 if any(m.kind == C.BIRD for m in movers) else 0) > 1:
            self.combos += 1
        xt = tight_x(statics, movers)
        if xt is not None and C.ROBUST:
            # Where a sensible pilot passes this obstacle: the middle +/- ROBUST_HALF of each gap
            # at its tightest point, keeping only altitudes from which the known world is still
            # passable (a combo can leave a dead-end pocket in the middle of a gap).
            kt = max(int(round(xt / C.CHECK_DX)), commit_k)
            k_end = max(int(math.ceil(max(fp.fx1 for fp in fps) / C.CHECK_DX)) + 1, kt)
            F = free_centre(kt * C.CHECK_DX, fps)
            V = viable_back(kt, k_end, fps, sched)
            P = _intersect(mid_band(F, C.ROBUST_HALF), V) or _intersect(F, V)
            if P:
                self.tight_k, self.P = kt, P
        self.R = R_commit
        self.commit_k = commit_k
        commit_x = commit_k * C.CHECK_DX
        self.open = [fp for fp in fps if fp.fx1 >= commit_x]
        self.next_start = xs + v * rng.uniform(gap_lo, gap_hi)


# ------------------------------------------------------------------------------------
# Independent grid-based verifier (used by tests)
# ------------------------------------------------------------------------------------
def verify_layout(hazards, schedule: Schedule, x_end: float, dx: float = 0.1, dy: float = 0.02):
    """Brute-force passability check on a raster, independent of the incremental sweep.

    Rasterises every footprint on an (x, y) grid, marks free corridors (vertical runs of free
    cells >= CORRIDOR_MIN), and propagates a reachable set of drone-centre cells left to right
    with the same climb rates. Returns (ok, fail_x, min_corridor_m).
    """
    fps = sorted((footprint_of(h) for h in hazards), key=lambda fp: fp.fx0)
    nxt = 0                     # sliding window over footprints sorted by left edge
    live = []
    xs = np.arange(C.DRONE_X, x_end, dx)
    ny = int(round(C.CEILING / dy)) + 1
    ys = np.arange(ny) * dy
    shrink = C.HALF_H + C.EDGE_MARGIN
    reach = np.abs(ys - C.START_Y) <= dy / 2 + 1e-9
    min_corr = math.inf
    for i, x in enumerate(xs):
        blocked = np.zeros(ny, dtype=bool)
        dd = False
        while nxt < len(fps) and fps[nxt].fx0 <= x:
            live.append(fps[nxt])
            nxt += 1
        live = [fp for fp in live if fp.fx1 >= x]
        for fp in live:
            if fp.fx0 <= x <= fp.fx1:
                if fp.kind == C.DOWNDRAFT:
                    dd = True
                b = fp.band(x)
                if b is not None:
                    blocked |= (ys >= b[0] - C.EDGE_MARGIN) & (ys <= b[1] + C.EDGE_MARGIN)
        # free runs
        free_c = np.zeros(ny, dtype=bool)
        best = 0.0
        j = 0
        while j < ny:
            if blocked[j]:
                j += 1
                continue
            k = j
            while k < ny and not blocked[k]:
                k += 1
            lo = 0.0 if j == 0 else ys[j]
            hi = C.CEILING if k == ny else ys[k - 1]
            if hi - lo >= C.CORRIDOR_MIN - 2 * dy:
                best = max(best, hi - lo)
                free_c |= (ys >= lo + shrink - dy) & (ys <= hi - shrink + dy) & (np.arange(ny) >= j) & (np.arange(ny) < k)
            j = k
        min_corr = min(min_corr, best)
        if i == 0:
            reach &= free_c | reach
            continue
        tau = dx / schedule.speed_at(x - C.DRONE_X)
        up = int(math.ceil((C.DOWNDRAFT_CLIMB_RATE if dd else C.CLIMB_RATE) * tau / dy))
        dn = int(math.ceil(C.CLIMB_RATE * tau / dy))
        grown = reach.copy()
        idx = np.flatnonzero(reach)
        if idx.size:
            # dilation by run: cells within [i-dn, i+up] of any reachable cell
            lo_i = np.maximum(idx - dn, 0)
            hi_i = np.minimum(idx + up, ny - 1)
            marks = np.zeros(ny + 1, dtype=np.int32)
            np.add.at(marks, lo_i, 1)
            np.add.at(marks, hi_i + 1, -1)
            grown = np.cumsum(marks[:-1]) > 0
        reach = grown & free_c
        if not reach.any():
            return False, float(x), min_corr
    return True, None, min_corr
