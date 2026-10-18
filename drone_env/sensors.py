"""Observation builder (SPEC §5): self vector, 13 rays, hazard tokens and masks.

The only randomness here is fog noise, drawn from the simulator's *sensor* RNG stream.
Nothing in this module changes the world.

Sensor specs (``sim.sensor_spec``; pilots get the spec they were made for):

* v2.3 (default): 20 m range, 13 values per hazard row. y_bottom / y_top are the body where it is
  now, vy comes from the tracker's last two sightings, and 5 tracker values follow: seconds watched
  / 2 (capped at 1), the lowest and highest body edge seen so far, and the predicted lowest and
  highest body edge while the drone passes it (``drone_env.tracker``; the band the teacher plans
  with, ``tracked_movers``). A static hazard is fully known: watched 1, seen = predicted = its
  bounds.
* v2.2: 16 m, 8 values; a moving hazard's y_bottom / y_top give the whole band its path covers
  (from the game's own path settings: pilots made with ``bird_envelope`` teachers).
* before v2.2: 16 m, 8 values, the body where it is now.
"""
from __future__ import annotations

import math

import numpy as np

from . import constants as C
from . import tracker
from .generator import BOX, mountain_segments, storm_ceiling_at, storm_segments, tent_max

_COS = np.array(C.RAY_COS, dtype=np.float64)
_SIN = np.array(C.RAY_SIN, dtype=np.float64)
_INV_COS = 1.0 / _COS                                   # all rays point forward (|angle| <= 60)
_SIN_SAFE = np.where(_SIN == 0.0, 1e-12, _SIN)
_INV_SIN = 1.0 / _SIN_SAFE
_GROUND_K = np.where(_SIN < 0.0, -1.0 / np.where(_SIN < 0.0, _SIN, -1.0), np.inf)   # t = y * k
_CEIL_K = np.where(_SIN > 0.0, 1.0 / np.where(_SIN > 0.0, _SIN, 1.0), np.inf)      # t = (10-y) * k
_EYE5 = np.eye(C.N_ACTIONS, dtype=np.float32)
_NR = C.NORM_RANGE


def _seg_point_dist(ax, ay, bx, by, px, py) -> float:
    ex, ey = bx - ax, by - ay
    L2 = ex * ex + ey * ey
    t = ((px - ax) * ex + (py - ay) * ey) / L2 if L2 > 0 else 0.0
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    dx, dy = ax + t * ex - px, ay + t * ey - py
    return math.sqrt(dx * dx + dy * dy)


def _rect_point_dist(x0, y0, x1, y1, px, py) -> float:
    dx = x0 - px if px < x0 else (px - x1 if px > x1 else 0.0)
    dy = y0 - py if py < y0 else (py - y1 if py > y1 else 0.0)
    return math.sqrt(dx * dx + dy * dy)


def cast_rays(X: float, Y: float, boxes: list, segs: list, rng_range: float) -> np.ndarray:
    """First-hit distance (metres, capped at rng_range) for the 13 rays from (X, Y)."""
    y = Y if Y > 1e-9 else 1e-9
    t = np.minimum(y * _GROUND_K, (C.CEILING - y) * _CEIL_K)
    if boxes:
        b = np.asarray(boxes, dtype=np.float64)                       # (B, 4) x0 y0 x1 y1
        tx0 = (b[:, 0] - X)[None, :] * _INV_COS[:, None]
        tx1 = (b[:, 2] - X)[None, :] * _INV_COS[:, None]
        ty0 = (b[:, 1] - y)[None, :] * _INV_SIN[:, None]
        ty1 = (b[:, 3] - y)[None, :] * _INV_SIN[:, None]
        tenter = np.maximum(tx0, np.minimum(ty0, ty1))
        texit = np.minimum(tx1, np.maximum(ty0, ty1))
        hit = (tenter <= texit) & (texit >= 0.0)
        tb = np.where(hit, np.maximum(tenter, 0.0), np.inf).min(axis=1)
        t = np.minimum(t, tb)
    if segs:
        s = np.asarray(segs, dtype=np.float64)                        # (S, 4) ax ay bx by
        ex = (s[:, 2] - s[:, 0])[None, :]
        ey = (s[:, 3] - s[:, 1])[None, :]
        apx = (s[:, 0] - X)[None, :]
        apy = (s[:, 1] - y)[None, :]
        c = _COS[:, None]
        sn = _SIN[:, None]
        denom = c * ey - sn * ex
        ok = np.abs(denom) > 1e-12
        den = np.where(ok, denom, 1.0)
        ts = (apx * ey - apy * ex) / den
        us = (apx * sn - apy * c) / den
        hit = ok & (ts >= 0.0) & (us >= 0.0) & (us <= 1.0)
        tseg = np.where(hit, ts, np.inf).min(axis=1)
        t = np.minimum(t, tseg)
    return np.minimum(t, rng_range)


def observe(sim) -> dict:
    X = sim.distance_m + C.DRONE_X
    Y = sim.y
    t = sim.frame * C.DT
    fog = sim.fog_end > sim.frame
    spec = tuple(getattr(sim, "sensor_spec", C.SENSOR_SPEC))
    R = C.FOG_RANGE if fog else C.sensor_range(spec)
    tail, nose = X - C.HALF_W, X + C.HALF_W
    v = sim.speed
    reach = nose + R
    band = spec == (2, 2)
    tracked = spec >= (2, 3)
    preds = tracker.predict(sim) if tracked else None

    boxes, segs, toks = [], [], []
    for h in sim.statics:
        if h.x0 > reach:
            break
        if h.x1 < tail:
            continue
        k = h.kind
        if k == C.BUILDING:
            boxes.append((h.x0, h.y0, h.x1, h.y1))
            dx_ = h.x0 - X if X < h.x0 else (X - h.x1 if X > h.x1 else 0.0)
            dy_ = h.y0 - Y if Y < h.y0 else (Y - h.y1 if Y > h.y1 else 0.0)
            dist = math.sqrt(dx_ * dx_ + dy_ * dy_)
        elif k == C.MOUNTAIN:
            sg = mountain_segments(h)
            segs.extend(sg)
            if h.x0 <= X <= h.x1 and Y <= tent_max(h, X, X):
                dist = 0.0
            else:
                dist = min(_seg_point_dist(*s, X, Y) for s in sg)
        elif k == C.STORM:
            sg = storm_segments(h)
            segs.extend(sg)
            if h.x0 <= X <= h.x1 and Y >= storm_ceiling_at(h, X):
                dist = 0.0
            else:
                dist = min(_seg_point_dist(*s, X, Y) for s in sg)
        elif h.parts:   # v2.0: boxes and wires (rays see a wire as a line)
            dist = math.inf
            for q in h.parts:
                if q[0] == BOX:
                    boxes.append(q[1:5])
                    d = _rect_point_dist(q[1], q[2], q[3], q[4], X, Y)
                else:
                    segs.append(q[1:5])
                    d = max(_seg_point_dist(q[1], q[2], q[3], q[4], X, Y) - q[5], 0.0)
                if d < dist:
                    dist = d
        else:  # DOWNDRAFT: a zone, rays pass through it
            dist = h.x0 - X if X < h.x0 else (X - h.x1 if X > h.x1 else 0.0)
        gap = h.x0 - nose
        toks.append((dist, h.uid, k, (0.0 if gap <= 0.0 else gap) / _NR, h.y0, h.y1, h.x1 - h.x0,
                     -v, 0.0, 0.0, 1.0, h.y0, h.y1, h.y0, h.y1))
    for m in sim.movers:
        x0 = m.xr - m.s * (t - m.tr)
        x1 = x0 + m.w
        if x1 < tail or x0 > reach:
            continue
        my = m.y_at(t)
        y0, y1 = my - m.h * 0.5, my + m.h * 0.5
        boxes.append((x0, y0, x1, y1))
        dx_ = x0 - X if X < x0 else (X - x1 if X > x1 else 0.0)
        dy_ = y0 - Y if Y < y0 else (Y - y1 if Y > y1 else 0.0)
        gap = x0 - nose
        uid = abs(m.uid)
        tk = (math.sqrt(dx_ * dx_ + dy_ * dy_), uid, m.kind, (0.0 if gap <= 0.0 else gap) / _NR)
        if tracked:
            pr = preds.get(uid)
            if pr is None:     # not tracked (a sim with tracking off): first sight
                pr = (0.0, y0, y1, y0, y1, 0.0)
            age, s_lo, s_hi, p_lo, p_hi, vy = pr[:6]
            toks.append(tk + (y0, y1, m.w, -(v + m.s), vy, 1.0,
                              min(age / C.TRACK_WATCH_NORM_S, 1.0), s_lo, s_hi, p_lo, p_hi))
        else:
            if band and not m.still:
                ty0, ty1 = m.base - m.down - m.h * 0.5, m.base + m.up + m.h * 0.5
            else:
                ty0, ty1 = y0, y1
            toks.append(tk + (ty0, ty1, m.w, -(v + m.s), m.vy_at(t), 1.0))

    # rays
    dist_m = cast_rays(X, Y, boxes, segs, R)
    rays = dist_m / _NR
    if fog:
        sim._rng_sensor = sim._rng_sensor.own()
        rays = rays + sim._rng_sensor.rng.normal(0.0, C.FOG_NOISE, C.N_RAYS)
        rays = np.clip(rays, 0.0, R / _NR)
    else:
        rays = np.clip(rays, 0.0, R / _NR)

    # hazard tokens: visible = nearest x-edge within range ahead, trailing edge not behind the tail
    ncol = C.hazard_cols(spec)
    hazards = np.zeros((C.MAX_HAZARDS, ncol), dtype=np.float32)
    htype = np.zeros(C.MAX_HAZARDS, dtype=np.int64)
    hmask = np.zeros(C.MAX_HAZARDS, dtype=np.int8)
    vis = [tk for tk in toks if tk[3] * _NR <= R]
    vis.sort(key=lambda tk: (tk[0], tk[1]))
    vis = vis[: C.MAX_HAZARDS]
    n = len(vis)
    if n:
        w = np.array([tk[6] for tk in vis])
        if tracked:
            hazards[:n] = np.array(
                [(tk[3], tk[4] / 10.0, tk[5] / 10.0, 0.0, tk[7] / _NR, tk[8] / 8.0, tk[0] / _NR, tk[9],
                  tk[10], tk[11] / 10.0, tk[12] / 10.0, tk[13] / 10.0, tk[14] / 10.0) for tk in vis],
                dtype=np.float32)
        else:
            hazards[:n] = np.array(
                [(tk[3], tk[4] / 10.0, tk[5] / 10.0, 0.0, tk[7] / _NR, tk[8] / 8.0, tk[0] / _NR, tk[9])
                 for tk in vis], dtype=np.float32)
        hazards[:n, 3] = np.minimum(w / 8.0, 1.5)
        htype[:n] = [tk[2] for tk in vis]
        hmask[:n] = 1
    sim.last_obs_uids = tuple(tk[1] for tk in vis)

    selfv = np.empty(11, dtype=np.float32)
    selfv[0] = Y / 10.0
    selfv[1] = sim.vy / 8.0
    selfv[2] = 1.0 if sim.tuck > 0 else 0.0
    selfv[3] = sim.cooldown * C.DT / 1.0
    selfv[4] = (v - C.SPEED_NORM_V0) / 8.0      # v1.7: (v - 8) / 8, L1 = 0 ... L5 = 1
    selfv[5] = (sim.gust_a + sim.dd_a) / 8.0
    selfv[6:] = _EYE5[sim.last_action]

    return {
        "self": selfv,
        "rays": rays.astype(np.float32),
        "hazards": hazards,
        "hazard_type": htype,
        "hazard_mask": hmask,
        "action_mask": sim.action_mask(),
    }
