"""DroneDash simulator: physics, hazards, collisions, interruptions (SPEC §2–4, §7).

Conventions
-----------
* World x grows in the flight direction. The drone centre is at ``X = distance_m + DRONE_X``.
* ``y`` is the centre of the drone hitbox. The hitbox is 0.9 m wide and 0.35 m tall
  (0.21 m tucked during a DUCK dive, shrinking around the centre). Ground crash: hitbox bottom <= 0.
  Ceiling crash: hitbox top >= 10.
* One ``step(action)`` = one frame of 1/30 s. No observation is built in ``step``.

RNG streams (all derived from ``numpy.random.SeedSequence(seed)``):
* world/layout -> obstacle generator
* world/events -> gusts, surges, fog banks
* sensor       -> fog noise only (used by ``observe()``)
Observing never touches the world streams, so an env and a bare Simulator driven by the same
actions stay identical. Clones share RNG objects copy-on-write.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import constants as C
from . import events
from . import tracker
from .generator import (NEVER, LayoutGenerator, Mover, RNGRef, Schedule, Static, mountain_segments,
                        storm_min, storm_segments, tent_max)

_MASK_ALL = np.ones(C.N_ACTIONS, dtype=np.int8)
_MASK_COOLDOWN = np.array([1, 1, 1, 0, 0], dtype=np.int8)   # JUMP and DUCK share one cooldown
_MASK_BURST = np.array([1, 0, 0, 0, 0], dtype=np.int8)      # during a jump or a dive

_CAUSE = {C.BUILDING: "building", C.MOUNTAIN: "mountain", C.BIRD: "bird", C.TRAFFIC: "traffic",
          C.STORM: "storm", C.PLANE: "plane"}


def _seg_point_dist(ax, ay, bx, by, px, py) -> float:
    ex, ey = bx - ax, by - ay
    L2 = ex * ex + ey * ey
    t = 0.0 if L2 == 0 else ((px - ax) * ex + (py - ay) * ey) / L2
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    dx, dy = ax + t * ex - px, ay + t * ey - py
    return math.sqrt(dx * dx + dy * dy)


def _rect_point_dist(x0, y0, x1, y1, px, py) -> float:
    dx = x0 - px if px < x0 else (px - x1 if px > x1 else 0.0)
    dy = y0 - py if py < y0 else (py - y1 if py > y1 else 0.0)
    return math.sqrt(dx * dx + dy * dy)


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return (o1 * o2 <= 0) and (o3 * o4 <= 0)


def _seg_rect_dist(ax, ay, bx, by, x0, y0, x1, y1) -> float:
    """Distance between a segment and an axis-aligned rectangle (0 if they touch)."""
    if (x0 <= ax <= x1 and y0 <= ay <= y1) or (x0 <= bx <= x1 and y0 <= by <= y1):
        return 0.0
    edges = ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0))
    for e in edges:
        if _segments_intersect(ax, ay, bx, by, *e):
            return 0.0
    d = min(_rect_point_dist(x0, y0, x1, y1, ax, ay), _rect_point_dist(x0, y0, x1, y1, bx, by))
    for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        d2 = _seg_point_dist(ax, ay, bx, by, px, py)
        if d2 < d:
            d = d2
    return d


def _rect_rect_dist(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1) -> float:
    dx = bx0 - ax1 if bx0 > ax1 else (ax0 - bx1 if ax0 > bx1 else 0.0)
    dy = by0 - ay1 if by0 > ay1 else (ay0 - by1 if ay0 > by1 else 0.0)
    return math.sqrt(dx * dx + dy * dy)


def parts_rect_dist(h: Static, x0, y0, x1, y1) -> float:
    """Distance from a rectangle to a parts-built static (boxes, and wires with their thickness);
    <= 0 means they touch."""
    best = math.inf
    for p in h.parts:
        if p[0] == 0:     # BOX
            d = _rect_rect_dist(x0, y0, x1, y1, p[1], p[2], p[3], p[4])
        else:             # WIRE
            d = _seg_rect_dist(p[1], p[2], p[3], p[4], x0, y0, x1, y1) - p[5]
        if d < best:
            best = d
            if best <= 0.0:
                return 0.0
    return best


def _first_block_x(h: Static, lo: float, hi: float, x_from: float):
    """First world x >= x_from where the static ``h`` blocks the height band [lo, hi], or None."""
    k = h.kind
    if h.parts:
        best = None
        for p in h.parts:
            if p[0] == 0:                                   # BOX
                if p[4] < lo or p[2] > hi or p[3] < x_from:
                    continue
                xb = p[1] if p[1] > x_from else x_from
            else:                                           # WIRE: where its line enters the band
                _, ax, ay, bx, by, r = p
                if ax > bx:
                    ax, ay, bx, by = bx, by, ax, ay
                if bx < x_from:
                    continue
                l2, h2 = lo - r, hi + r
                if bx - ax < 1e-9 or abs(by - ay) < 1e-12:
                    if max(ay, by) < l2 or min(ay, by) > h2:
                        continue
                    xb = ax
                else:
                    ta, tb = (l2 - ay) / (by - ay), (h2 - ay) / (by - ay)
                    if ta > tb:
                        ta, tb = tb, ta
                    ta, tb = max(ta, 0.0), min(tb, 1.0)
                    if ta > tb:
                        continue
                    xa, xe = ax + ta * (bx - ax), ax + tb * (bx - ax)
                    if xe < x_from:
                        continue
                    xb = xa
                xb = xb if xb > x_from else x_from
            if best is None or xb < best:
                best = xb
        return best
    if k == C.BUILDING:
        if h.y1 < lo or h.x1 < x_from:
            return None
        return h.x0 if h.x0 > x_from else x_from
    if k == C.MOUNTAIN:
        if h.y1 < lo:
            return None
        xb = h.x0 + (h.xc - h.x0) * (lo / h.y1 if lo > 0.0 else 0.0)
        return xb if xb > x_from else (x_from if x_from <= h.x1 - (h.x1 - h.xc) * max(lo, 0.0) / h.y1 else None)
    if k == C.STORM:
        if hi < h.y0:
            return None
        depth = C.CEILING - h.y0
        xb = h.x0 + h.r * (C.CEILING - hi) / depth if depth > 0 else h.x0
        x_end = h.x1 - h.r * (C.CEILING - hi) / depth if depth > 0 else h.x1
        if x_end < x_from:
            return None
        return xb if xb > x_from else x_from
    return None


class Simulator:
    """One DroneDash world. See module docstring for conventions."""

    def __init__(self, level: str = "L1", seed: int = 0, generate: bool = True, record: bool = False):
        self.level = level
        self.seed = int(seed)
        self.schedule = Schedule(level)
        ss = np.random.SeedSequence(self.seed)
        s_layout, s_events, s_sensor = ss.spawn(3)
        self._rng_events = RNGRef(np.random.Generator(np.random.PCG64(s_events)))
        self._rng_sensor = RNGRef(np.random.Generator(np.random.PCG64(s_sensor)))
        self.gen = LayoutGenerator(self.schedule, np.random.Generator(np.random.PCG64(s_layout)),
                                   record=record)
        # drone
        self.y = C.START_Y
        self.vy = C.START_VY
        self.burst = 0           # jump/dive burst frames left
        self.cooldown = 0        # shared JUMP/DUCK cooldown frames left
        self.tuck = 0            # tucked (dive) frames left; a dive is tucked for its whole burst
        self.v_target = 0.0
        self.last_action = C.HOLD
        # progress
        self.frame = 0
        self.distance_m = 0.0
        self.crashed = False
        self.crash_cause = None
        self.masked_count = 0
        self.cleared_total = 0
        self.cleared_step = 0
        # interruptions
        self.calm = False
        self.gust_a = 0.0
        self.gust_end = 0
        self.surge_end = 0
        self.fog_end = 0
        self.next_gust = self.next_surge = self.next_fog = NEVER
        self._ev_idx = 0
        # world
        self.statics = []        # Static, sorted by x0
        self.movers = []         # active Mover
        self.dormant = []        # dormant Mover, in generation order
        self.pending = []        # uncleared solid statics
        self.generate = generate
        self.sensor_spec = C.SENSOR_SPEC   # what observe() builds; older pilots get their own (SPEC v2.2/v2.3)
        self.tracking = True     # v2.3 object tracker on (planner clones switch it off)
        self.tracks = {}         # v2.3: uid -> tracker.Track of every mover the sensors show
        self._pred_cache = None
        self._lookahead = C.LOOKAHEAD
        self.dd_a = 0.0          # downdraft acceleration at the drone (felt next step)
        self.speed = self.schedule.speed_at(0.0)
        self.last_obs_uids = ()  # viewer-only: uid per hazard row of the last observe()
        events.reschedule(self)
        if generate:
            self._refill(force=True)
        self._update_local()
        tracker.update(self)

    # ------------------------------------------------------------------ properties ---
    @property
    def t(self) -> float:
        return self.frame * C.DT

    @property
    def x(self) -> float:
        """World x of the drone centre."""
        return self.distance_m + C.DRONE_X

    @property
    def tucked(self) -> bool:
        return self.tuck > 0

    @property
    def half_h(self) -> float:
        return C.HALF_H_TUCK if self.tuck > 0 else C.HALF_H

    @property
    def gust_active(self) -> bool:
        return self.gust_a != 0.0

    @property
    def surge_active(self) -> bool:
        return self.surge_end > self.frame

    @property
    def fog_active(self) -> bool:
        return self.fog_end > self.frame

    @property
    def a_wind(self) -> float:
        """Felt wind: gust + downdraft at the drone, acting on the next step."""
        return self.gust_a + self.dd_a

    @property
    def level_now(self) -> str:
        return self.schedule.spec_at(self.distance_m).name

    # ------------------------------------------------------------------ actions -------
    def action_mask(self) -> np.ndarray:
        if self.burst > 0 or self.tuck > 0:
            return _MASK_BURST.copy()
        if self.cooldown > 0:
            return _MASK_COOLDOWN.copy()
        return _MASK_ALL.copy()

    def available_actions(self) -> tuple:
        """Fast tuple form of action_mask() for planners."""
        if self.burst > 0 or self.tuck > 0:
            return (0,)
        if self.cooldown > 0:
            return (0, 1, 2)
        return (0, 1, 2, 3, 4)

    # ------------------------------------------------------------------ stepping ------
    def step(self, action: int) -> bool:
        """Advance one frame. Masked actions become HOLD (counted). Returns True on crash."""
        if self.crashed:
            return True
        a = int(action)
        vy = self.vy
        if self.burst > 0 or self.tuck > 0:
            # inside a jump or a dive: no control, the burst target holds (masked -> HOLD)
            if a != C.HOLD:
                self.masked_count += 1
                a = C.HOLD
            vt = C.DUCK_V if self.tuck > 0 else C.JUMP_V
        elif a == C.HOLD:
            vt = C.V_HOLD
        elif a == C.ASCEND:
            vt = C.V_ASCEND
        elif a == C.DESCEND:
            vt = C.V_DESCEND
        elif a == C.JUMP or a == C.DUCK:
            if self.cooldown > 0:
                self.masked_count += 1
                a = C.HOLD
                vt = C.V_HOLD
            elif a == C.JUMP:
                vy = C.JUMP_V
                self.burst = C.JUMP_BURST
                self.cooldown = C.JUMP_COOLDOWN
                vt = C.JUMP_V
            else:   # DUCK = dive: fast drop, tucked hitbox, same cooldown as JUMP
                vy = C.DUCK_V
                self.burst = C.DUCK_BURST
                self.tuck = C.DUCK_BURST
                self.cooldown = C.JUMP_COOLDOWN
                vt = C.DUCK_V
        else:
            raise ValueError(f"invalid action {action!r}")
        self.last_action = a
        self.v_target = vt

        # vertical physics: vy += (a_ctrl + a_wind) dt, then y += vy dt
        a_ctrl = C.CTRL_GAIN * (vt - vy)
        if a_ctrl > C.A_MAX:
            a_ctrl = C.A_MAX
        elif a_ctrl < -C.A_MAX:
            a_ctrl = -C.A_MAX
        vy += (a_ctrl + self.gust_a + self.dd_a) * C.DT
        self.vy = vy
        self.y += vy * C.DT

        # scroll
        self.distance_m += self.speed * C.DT
        self.frame += 1

        # collision at the new state (tucked hitbox if the tuck covers this frame)
        self._collide()

        # timers
        if self.burst > 0:
            self.burst -= 1
        if self.cooldown > 0:
            self.cooldown -= 1
        if self.tuck > 0:
            self.tuck -= 1

        events.update(self)
        if self.generate:
            self._refill()
        self._update_local()
        if self.tracking:
            tracker.update(self)
        return self.crashed

    def _collide(self) -> None:
        y = self.y
        hh = C.HALF_H_TUCK if self.tuck > 0 else C.HALF_H
        bottom, top = y - hh, y + hh
        if bottom <= 0.0:
            self.crashed, self.crash_cause = True, "ground"
            return
        if top >= C.CEILING:
            self.crashed, self.crash_cause = True, "ceiling"
            return
        X = self.distance_m + C.DRONE_X
        tail, nose = X - C.HALF_W, X + C.HALF_W
        for h in self.statics:
            if h.x0 > nose:
                break
            if h.x1 < tail:
                continue
            k = h.kind
            if k == C.BUILDING:
                if bottom <= h.y1:
                    break_cause = "building"
                    self.crashed, self.crash_cause = True, break_cause
                    return
            elif k == C.MOUNTAIN:
                if bottom <= tent_max(h, tail, nose):
                    self.crashed, self.crash_cause = True, "mountain"
                    return
            elif k == C.STORM:
                if top >= storm_min(h, tail, nose):
                    self.crashed, self.crash_cause = True, "storm"
                    return
            elif h.parts:
                if parts_rect_dist(h, tail, bottom, nose, top) <= 0.0:
                    self.crashed, self.crash_cause = True, C.HAZARD_NAMES[k]
                    return
        t = self.frame * C.DT
        for m in self.movers:
            x0 = m.xr - m.s * (t - m.tr)
            if x0 > nose or x0 + m.w < tail:
                continue
            my = m.y_at(t)
            hm = m.h * 0.5
            if bottom <= my + hm and top >= my - hm:
                self.crashed, self.crash_cause = True, _CAUSE[m.kind]
                return

    def _update_local(self) -> None:
        """After a step: clearing, mover activation, culling, felt downdraft, current speed."""
        X = self.distance_m + C.DRONE_X
        tail = X - C.HALF_W
        t = self.frame * C.DT
        # speed for the next step (nominal at this distance, surge if active)
        v = self.schedule.speed_at(self.distance_m)
        if self.surge_end > self.frame:
            v *= C.SURGE_MULT
        self.speed = v
        # activate movers whose meeting point is LOOKAHEAD ahead
        if self.dormant and self.dormant[0].xe - C.LOOKAHEAD <= X:
            nose = X + C.HALF_W
            keep = []
            for m in self.dormant:
                if m.xe - C.LOOKAHEAD <= X:
                    self.movers.append(m.activated(nose, v, t))
                else:
                    keep.append(m)
            self.dormant = keep
        # clearing reward bookkeeping
        cleared = 0
        if self.pending:
            p = self.pending
            if p[0].x0 < tail:
                keep = []
                for i, h in enumerate(p):
                    if h.x0 >= tail:
                        keep.extend(p[i:])
                        break
                    if h.x1 < tail:
                        cleared += 1
                    else:
                        keep.append(h)
                self.pending = keep
        for m in self.movers:
            if m.uid > 0 and (m.xr - m.s * (t - m.tr)) + m.w < tail:
                cleared += 1
        if cleared and self.movers:
            # mark cleared movers by negating uid in a fresh object list (objects stay immutable)
            new = []
            for m in self.movers:
                if m.uid > 0 and (m.xr - m.s * (t - m.tr)) + m.w < tail:
                    m = m.copy(uid=-m.uid)
                new.append(m)
            self.movers = new
        self.cleared_step = cleared
        self.cleared_total += cleared
        # felt downdraft at the drone centre
        dd = 0.0
        for h in self.statics:
            if h.x0 > X:
                break
            if h.kind == C.DOWNDRAFT and h.x1 >= X:
                dd = C.DOWNDRAFT_ACCEL
                break
        self.dd_a = dd
        # cull hazards that left the screen (every 15 frames)
        if self.frame % 15 == 0:
            cull = self.distance_m - 2.0
            if self.statics and self.statics[0].x1 < cull:
                self.statics = [h for h in self.statics if h.x1 >= cull]
            if self.movers:
                self.movers = [m for m in self.movers if (m.xr - m.s * (t - m.tr)) + m.w >= cull]

    def _refill(self, force: bool = False) -> None:
        X = self.distance_m + C.DRONE_X
        if force or self.gen.next_start < X + self._lookahead:
            n0 = len(self.statics)
            self.gen.generate_until(X + max(self._lookahead, C.LOOKAHEAD) + C.BATCH, self.statics,
                                    self.dormant)
            for h in self.statics[n0:]:
                if h.kind != C.DOWNDRAFT:
                    self.pending.append(h)

    # ------------------------------------------------------------------ planner API ---
    def clone(self, calm: bool = True, hold_gust: bool = False) -> "Simulator":
        """Fast independent copy.

        calm=True: the active gust and surge are removed and no new random interruptions start
        in the copy (SPEC §7). hold_gust=True (only with calm) instead keeps the gust the drone
        feels right now at its current strength for the whole life of the copy, i.e. the planner
        assumes the wind it can feel persists. Nothing about its remaining duration is revealed.
        """
        c = Simulator.__new__(Simulator)
        d = self.__dict__.copy()
        self._rng_events.shared = True
        self._rng_sensor.shared = True
        d["statics"] = self.statics[:]
        d["movers"] = self.movers[:]
        d["dormant"] = self.dormant[:]
        d["pending"] = self.pending[:]
        d["gen"] = self.gen.clone()
        if calm:
            # Planner copies refill the world only when it gets within 25 m (still beyond the
            # 20 m sensor range), so a 1 s plan never pays for generation. calm=False copies keep
            # the original threshold and reproduce the original exactly. Planner copies do not
            # track (the teacher plans with what the original's tracker knows now).
            d["_lookahead"] = C.CLONE_LOOKAHEAD
            d["tracking"] = False
        c.__dict__ = d
        if calm and not self.calm:      # a clone of a calm sim is already calm: exact copy
            gust = c.gust_a
            events.make_calm(c)
            if hold_gust and gust != 0.0:
                c.gust_a = gust
                c.gust_end = NEVER
            c.speed = c.schedule.speed_at(c.distance_m)
        return c

    def clearance(self) -> float:
        """Metres from the drone hitbox to the nearest hazard, the ground or the ceiling."""
        hh = C.HALF_H_TUCK if self.tuck > 0 else C.HALF_H
        y = self.y
        bx0, by0 = self.distance_m + C.DRONE_X - C.HALF_W, y - hh
        bx1, by1 = bx0 + C.HIT_W, y + hh
        best = by0 if by0 < C.CEILING - by1 else C.CEILING - by1
        if best <= 0.0:
            return 0.0
        for h in self.statics:
            if h.x0 > bx1 + best:
                break
            if h.x1 < bx0 - best or h.kind == C.DOWNDRAFT:
                continue
            k = h.kind
            if k == C.BUILDING:
                d = _rect_rect_dist(bx0, by0, bx1, by1, h.x0, h.y0, h.x1, h.y1)
            elif k == C.MOUNTAIN:
                if by0 <= tent_max(h, bx0, bx1):
                    return 0.0
                d = min(_seg_rect_dist(*s, bx0, by0, bx1, by1) for s in mountain_segments(h))
            elif h.parts:
                d = parts_rect_dist(h, bx0, by0, bx1, by1)
            else:  # STORM
                if by1 >= storm_min(h, bx0, bx1):
                    return 0.0
                d = min(_seg_rect_dist(*s, bx0, by0, bx1, by1) for s in storm_segments(h))
            if d < best:
                best = d
                if best <= 0.0:
                    return 0.0
        t = self.frame * C.DT
        for m in self.movers:
            x0 = m.xr - m.s * (t - m.tr)
            if x0 > bx1 + best or x0 + m.w < bx0 - best:
                continue
            my = m.y_at(t)
            d = _rect_rect_dist(bx0, by0, bx1, by1, x0, my - m.h / 2, x0 + m.w, my + m.h / 2)
            if d < best:
                best = d
        return max(best, 0.0)

    def path_time(self, cap_s: float, margin: float = 0.3) -> float:
        """Seconds until the drone, flying level at its current height (hitbox + ``margin`` above
        and below), would reach the first hazard in that band; ``cap_s`` if none within it. Static
        hazards close at the scroll speed, movers at scroll speed + their own speed (birds at their
        current height; a planner clone with bird_envelope or tracked_movers sees their band). Planner use."""
        v = self.speed if self.speed > 0.1 else 0.1
        hh = C.HALF_H_TUCK if self.tuck > 0 else C.HALF_H
        lo, hi = self.y - hh - margin, self.y + hh + margin
        X = self.distance_m + C.DRONE_X
        tail, nose = X - C.HALF_W, X + C.HALF_W
        best = cap_s
        reach = nose + v * cap_s
        for h in self.statics:
            if h.x0 > reach:
                break
            if h.x1 < tail or h.kind == C.DOWNDRAFT:
                continue
            xb = _first_block_x(h, lo, hi, nose)
            if xb is not None:
                t = (xb - nose) / v
                if t < best:
                    best = t if t > 0.0 else 0.0
        t_now = self.frame * C.DT
        for m in self.movers:
            x0 = m.xr - m.s * (t_now - m.tr)
            if x0 + m.w < tail:
                continue
            my = m.y_at(t_now)
            if my - m.h / 2 > hi or my + m.h / 2 < lo:
                continue
            close = v + m.s
            if close <= 0.1:
                continue
            t = (x0 - nose) / close
            if t < best:
                best = t if t > 0.0 else 0.0
        return best

    def observe(self) -> dict:
        from .sensors import observe
        return observe(self)

    # ------------------------------------------------------------------ geometry ------
    def hazard_bounds(self):
        """Viewer/sensor helper: list of (hazard, kind, x0, x1, y0, y1) at the current time."""
        t = self.frame * C.DT
        out = []
        for h in self.statics:
            out.append((h, h.kind, h.x0, h.x1, h.y0, h.y1))
        for m in self.movers:
            x0 = m.xr - m.s * (t - m.tr)
            my = m.y_at(t)
            out.append((m, m.kind, x0, x0 + m.w, my - m.h / 2, my + m.h / 2))
        return out

    # ------------------------------------------------------------------ test hooks ----
    def place_static(self, kind: int, x0: float, x1: float, y0: float, y1: float, parts=()) -> Static:
        """Insert a static hazard at world x (tests and scripted scenarios only). A v2.0 kind takes
        its ``parts`` (generator.BOX / WIRE tuples); x0..y1 are then its bounds."""
        uid = 10_000_000 + len(self.statics) + 1
        if parts:
            h = Static(uid, kind, x0, x1, y0, y1, xc=(x0 + x1) / 2, parts=tuple(parts))
        elif kind == C.MOUNTAIN:
            h = Static(uid, kind, x0, x1, 0.0, y1, xc=(x0 + x1) / 2)
        elif kind == C.STORM:
            h = Static(uid, kind, x0, x1, y0, C.CEILING, r=(x1 - x0) * C.STORM_RAMP_FRAC)
        else:
            h = Static(uid, kind, x0, x1, y0, y1)
        self.statics = sorted(self.statics + [h], key=lambda s: s.x0)
        if kind != C.DOWNDRAFT:
            self.pending = sorted(self.pending + [h], key=lambda s: s.x0)
        self._update_local_wind()
        return h

    def place_mover(self, kind: int, x0: float, y: float, w: float, h: float, s: float = 0.0,
                    amp: float = 0.0, period: float = 1.5, phase: float = 0.0, sub: int = 0, **path) -> Mover:
        """Insert an active moving hazard whose left edge is at world x0 now (tests only). ``path``:
        optional v2.3 Mover fields (fm, omega_m, phase_m, dive, omega_d, phase_d)."""
        uid = 20_000_000 + len(self.movers) + 1
        m = Mover(uid, kind, sub, w, h, y, amp, 2 * math.pi / period, phase, s, x0, x0, self.frame * C.DT, **path)
        self.movers = self.movers + [m]
        tracker.update(self)
        return m

    def _update_local_wind(self) -> None:
        X = self.x
        self.dd_a = 0.0
        for h in self.statics:
            if h.kind == C.DOWNDRAFT and h.x0 <= X <= h.x1:
                self.dd_a = C.DOWNDRAFT_ACCEL

    # ------------------------------------------------------------------ hashing -------
    def state_tuple(self) -> tuple:
        t = self.frame * C.DT
        stat = tuple(h.key() for h in self.statics)
        mov = tuple(m.key() for m in self.movers)
        dor = tuple(m.key() for m in self.dormant)
        ev = self._rng_events.rng.bit_generator.state["state"]["state"]
        lay = self.gen.rng.rng.bit_generator.state["state"]["state"]
        return (self.level, self.seed, self.frame, round(self.y, 10), round(self.vy, 10),
                round(self.distance_m, 9), self.burst, self.cooldown, self.tuck, self.last_action,
                self.crashed, self.crash_cause, self.masked_count, self.cleared_total,
                round(self.gust_a, 10), self.gust_end, self.surge_end, self.fog_end, self.next_gust,
                self.next_surge, self.next_fog, round(t, 9), stat, mov, dor, ev, lay,
                round(self.gen.next_start, 9))

    def state_hash(self) -> str:
        return hashlib.sha256(repr(self.state_tuple()).encode()).hexdigest()
