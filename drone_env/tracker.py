"""Object tracker (SPEC v2.3 §5): what a real drone's sensing software can know about a bird, a
drone, a balloon or a plane, which is only what its sensors have seen of it so far.

* ``update(sim)`` (every step, from the simulator) records the centre height of every mover the
  sensors show now (leading edge within nose + range, 8 m in fog; trailing edge not behind the
  tail). A mover that drops out of view loses its track; seen again, it starts a new one.
* ``predict(sim)`` gives, per tracked mover: seconds watched, the lowest / highest body edge seen,
  the predicted lowest / highest body edge while the drone passes it, its vertical speed from the
  last two sightings, and the best-guess band without margins. The prediction fits a sine to the last ``keep_s`` of sightings (rhythm
  searched over the type's range; the last height seen while there are too few sightings), evaluates it
  across the pass window, widens it by calibrated margins (by watched time and how far ahead the
  pass window ends, interpolated between the table's cells; ``configs/tracker.yaml``) and keeps it within what the type can reach from what
  was seen (its largest top-to-bottom span). Planes never move up or down (span 0).

Nothing here reads a mover's hidden path settings (base, amplitude, rhythm, dive): only heights
seen at sighting times, the mover's type (birds, drones and planes move differently), and the
type's largest span and rhythm range from game.yaml (what a real system knows about a kind of
object). Tracks are immutable tuples, so clones share them for free.
"""
from __future__ import annotations

import bisect
import math

import numpy as np
import yaml

from . import constants as C

# type groups: margins and limits per group
G_BIRD, G_TRAFFIC, G_STILL = 0, 1, 2
GROUP = {C.BIRD: G_BIRD, C.TRAFFIC: G_TRAFFIC, C.PLANE: G_STILL}
GROUP_NAMES = ("bird", "traffic", "still")

_b, _t = C.HZ["bird"], C.HZ["traffic"]
# largest top-to-bottom span of a mover's centre path (hard limit) and largest swing amplitude
SPAN = (2.0 * _b["amplitude"][1] + _b["dive_depth"][1], 2.0 * _t["drift_amplitude"][1], 0.0)
AMP_MAX = (float(_b["amplitude"][1]), float(_t["drift_amplitude"][1]), 0.0)


def _omega_grid(periods, fm_max, fm_period_min, n):
    """Angular frequencies the fit tries: the type's period range, widened by its rhythm wander."""
    wander = fm_max * 2 * math.pi / fm_period_min
    lo = max(2 * math.pi / periods[1] - wander, 0.3)
    hi = 2 * math.pi / periods[0] + wander
    return np.geomspace(lo, hi, n)


OMEGAS = (_omega_grid(_b["period_s"], _b["rhythm_change"][1], _b["rhythm_period_s"][0], 16),
          _omega_grid(_t["drift_period_s"], _t["rhythm_change"][1], _t["rhythm_period_s"][0], 12),
          None)

KEEP = max(int(round(C.TRACK_KEEP_S * C.FPS)), 2)

# margin table bins: seconds watched, and seconds ahead to the end of the pass window
AGE_EDGES = (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7)            # 10 bins
AHEAD_EDGES = (0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2)                # 9 bins


def _mids(edges):
    e = (0.0,) + tuple(edges)
    return np.array([(a + b) / 2 for a, b in zip(e, e[1:])] + [e[-1] + (e[-1] - e[-2]) / 2])


AGE_MIDS, AHEAD_MIDS = _mids(AGE_EDGES), _mids(AHEAD_EDGES)   # where each cell's margin applies exactly


class Track:
    """Sightings of one mover: first sighting time, the last KEEP (time, centre height) pairs and
    the lowest / highest centre seen over the whole track. Immutable."""

    __slots__ = ("t0", "ts", "ys", "lo", "hi")

    def __init__(self, t0, ts, ys, lo, hi):
        self.t0, self.ts, self.ys, self.lo, self.hi = t0, ts, ys, lo, hi

    @staticmethod
    def start(t: float, y: float) -> "Track":
        return Track(t, (t,), (y,), y, y)

    def add(self, t: float, y: float) -> "Track":
        ts, ys = self.ts, self.ys
        if t <= ts[-1]:                   # seen again in the same frame (a scripted insert): no new sighting
            return self
        if len(ts) >= KEEP:
            ts, ys = ts[1 - KEEP:], ys[1 - KEEP:]
        return Track(self.t0, ts + (t,), ys + (y,), y if y < self.lo else self.lo, y if y > self.hi else self.hi)


def in_view(sim, m, t, R):
    """Left edge x of mover m at time t if the sensors show it (range R from the nose), else None."""
    x0 = m.xr - m.s * (t - m.tr)
    X = sim.distance_m + C.DRONE_X
    if x0 > X + C.HALF_W + R or x0 + m.w < X - C.HALF_W:
        return None
    return x0


def view_range(sim) -> float:
    spec = getattr(sim, "sensor_spec", C.SENSOR_SPEC)
    return C.FOG_RANGE if sim.fog_end > sim.frame else C.sensor_range(spec)


def update(sim) -> None:
    """Record the movers the sensors show now; forget the rest."""
    old = sim.tracks
    t = sim.frame * C.DT
    R = view_range(sim)
    new = {}
    for m in sim.movers:
        if in_view(sim, m, t, R) is None:
            continue
        uid = -m.uid if m.uid < 0 else m.uid
        y = m.y_at(t)
        tr = old.get(uid)
        new[uid] = Track.start(t, y) if tr is None else tr.add(t, y)
    sim.tracks = new


# ------------------------------------------------------------------------------------------
# prediction
# ------------------------------------------------------------------------------------------
def _fit(tau: np.ndarray, y: np.ndarray, omegas: np.ndarray, amp_max: float):
    """Least-squares y ~ c + a sin(w tau) + b cos(w tau) for each w; the best w. Returns (c, a, b, w)."""
    n = len(tau)
    W = omegas[:, None] * tau[None, :]
    s, c = np.sin(W), np.cos(W)
    A = np.empty((len(omegas), 3, 3))
    A[:, 0, 0] = n
    A[:, 0, 1] = A[:, 1, 0] = s.sum(1)
    A[:, 0, 2] = A[:, 2, 0] = c.sum(1)
    A[:, 1, 1] = (s * s).sum(1) + 1e-6
    A[:, 1, 2] = A[:, 2, 1] = (s * c).sum(1)
    A[:, 2, 2] = (c * c).sum(1) + 1e-6
    rhs = np.stack([np.full(len(omegas), y.sum()), s @ y, c @ y], 1)
    x = np.linalg.solve(A, rhs[..., None])[..., 0]
    sse = float(y @ y) - (x * rhs).sum(1)
    j = int(np.argmin(sse))
    c0, a, b = x[j]
    amp = math.hypot(a, b)
    w = float(omegas[j])
    if amp > amp_max:                     # an implausible swing: keep its timing, cap its size, refit the centre
        k = amp_max / amp
        a, b = a * k, b * k
        c0 = float(np.mean(y - a * s[j] - b * c[j]))
    return float(c0), float(a), float(b), w


def core_band(tr: Track, t_now: float, ta: float, tb: float, group: int) -> tuple[float, float]:
    """Centre heights the track's best guess covers over the pass window [ta, tb] seconds ahead
    (before margins and limits)."""
    if group == G_STILL:
        return tr.ys[-1], tr.ys[-1]
    n = len(tr.ts)
    if n < C.TRACK_MIN_FIT:
        return tr.ys[-1], tr.ys[-1]
    tau = np.asarray(tr.ts) - t_now
    y = np.asarray(tr.ys)
    c0, a, b, w = _fit(tau, y, OMEGAS[group], AMP_MAX[group])
    q = np.linspace(ta, tb, C.TRACK_WINDOW_POINTS) * w
    yy = c0 + a * np.sin(q) + b * np.cos(q)
    return float(yy.min()), float(yy.max())


def _bin(edges, v):
    return bisect.bisect_right(edges, v)


_MARGINS = None


def margins():
    """(3, 10, 9, 2) calibrated margins (down, up) in metres by group, watched-time bin and
    ahead bin; ``configs/tracker.yaml`` (scripts/calibrate_tracker.py). Before calibration: the
    type's whole span (the prediction is then just "within reach of what was seen")."""
    global _MARGINS
    if _MARGINS is None:
        m = np.zeros((3, len(AGE_EDGES) + 1, len(AHEAD_EDGES) + 1, 2))
        if C.TRACKER_YAML.exists():
            raw = yaml.safe_load(C.TRACKER_YAML.read_text())
            for g, name in enumerate(GROUP_NAMES[:2]):
                m[g, :, :, 0] = np.asarray(raw["margins"][name]["down"])
                m[g, :, :, 1] = np.asarray(raw["margins"][name]["up"])
        else:
            for g in (G_BIRD, G_TRAFFIC):
                m[g] = SPAN[g]
        _MARGINS = m
    return _MARGINS


def _lerp_at(mids, v):
    """(lower index, weight of the upper one) for v on the cell midpoints, clamped at the ends."""
    j = int(np.searchsorted(mids, v))
    if j <= 0:
        return 0, 0.0
    if j >= len(mids):
        return len(mids) - 2, 1.0
    return j - 1, (v - mids[j - 1]) / (mids[j] - mids[j - 1])


def margin(group: int, age: float, tb: float) -> np.ndarray:
    """(down, up) margin in metres, interpolated between the calibrated cells (no steps at the
    cells' edges, so a band does not jump as the drone closes in)."""
    m = margins()[group]
    i, a = _lerp_at(AGE_MIDS, age)
    j, b = _lerp_at(AHEAD_MIDS, tb)
    return ((1 - a) * ((1 - b) * m[i, j] + b * m[i, j + 1]) + a * ((1 - b) * m[i + 1, j] + b * m[i + 1, j + 1]))


def finish(core: tuple[float, float], tr: Track, age: float, tb: float, group: int,
           scale: float = 1.0) -> tuple[float, float]:
    """Core band widened by the calibrated margins (times ``scale``) and kept within the type's reach
    from what was seen: every height of its path lies within SPAN of every other."""
    lo, hi = core
    mg = margin(group, age, tb)
    lo, hi = lo - scale * mg[0], hi + scale * mg[1]
    b_lo, b_hi = tr.hi - SPAN[group], tr.lo + SPAN[group]
    lo = b_lo if lo < b_lo else (b_hi if lo > b_hi else lo)
    hi = b_lo if hi < b_lo else (b_hi if hi > b_hi else hi)
    return lo, (hi if hi > lo else lo)


def window(sim, m, x0: float) -> tuple[float, float]:
    """Seconds from now until the drone's nose reaches mover m (0 if overlapping) and until its
    tail has passed it, at the current closing speed."""
    nose = sim.distance_m + C.DRONE_X + C.HALF_W
    close = sim.speed + m.s
    close = close if close > 0.5 else 0.5
    gap = x0 - nose
    return (gap if gap > 0.0 else 0.0) / close, (gap + m.w + C.HIT_W) / close


def predict(sim) -> dict:
    """uid -> (watched s, seen low, seen high, predicted low, predicted high, vy, best-guess low,
    best-guess high) for every tracked mover the sensors show now; heights are body edges (centre
    -/+ half its height); the best guess is the band without margins. Cached per frame and track set."""
    cache = sim._pred_cache
    if cache is not None and cache[0] == sim.frame and cache[1] is sim.tracks:
        return cache[2]
    tracks = sim.tracks
    t = sim.frame * C.DT
    R = view_range(sim)
    out = {}
    for m in sim.movers:
        uid = -m.uid if m.uid < 0 else m.uid
        tr = tracks.get(uid)
        if tr is None:
            continue
        x0 = in_view(sim, m, t, R)
        if x0 is None:
            continue
        g = GROUP.get(m.kind, G_STILL)
        ta, tb = window(sim, m, x0)
        age = t - tr.t0
        core = core_band(tr, t, ta, tb, g)
        lo, hi = finish(core, tr, age, tb, g)
        c_lo, c_hi = finish(core, tr, age, tb, g, scale=0.0)
        n = len(tr.ts)
        vy = (tr.ys[-1] - tr.ys[-2]) / (tr.ts[-1] - tr.ts[-2]) if n >= 2 else 0.0
        hh = m.h * 0.5
        out[uid] = (age, tr.lo - hh, tr.hi + hh, lo - hh, hi + hh, vy, c_lo - hh, c_hi + hh)
    sim._pred_cache = (sim.frame, tracks, out)
    return out
