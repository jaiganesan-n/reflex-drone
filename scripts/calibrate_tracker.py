"""Calibrate the v2.3 object tracker's margins (SPEC §5) -> configs/tracker.yaml.

    uv run python scripts/calibrate_tracker.py [--birds 20000] [--traffic 10000] [--seed 0]

Samples birds and drones / balloons from the generator (game.yaml settings), flies each past the
drone the way the game does (first seen at the sensor range, closing at the scroll speed plus its
own speed; 10 or 12 m/s, sometimes in a surge or in fog), and at every frame compares the tracker's
best-guess band for the pass window (``tracker.core_band``) with where the mover's centre really
goes in that window. Per type, watched-time bin and ahead bin, the margin below (above) is the
ahead bin's quantile (game.yaml ``quantile_by_ahead``) of how far it went below (above) the guess,
at least 0.02 m; empty cells take the nearest visited cell of their row; younger tracks never get
smaller margins than older ones, and longer look-aheads never smaller than shorter ones. Then it
reports how often the finished band (``tracker.finish``) holds the mover and how wide it is.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drone_env import constants as C          # noqa: E402
from drone_env import tracker as T            # noqa: E402
from drone_env.sim import Simulator           # noqa: E402

N_AGE, N_AHEAD = len(T.AGE_EDGES) + 1, len(T.AHEAD_EDGES) + 1


def sightings(m, rng, samples):
    """Fly mover m past the drone; append (age, ahead, err_down, err_up, centre lo/hi truth, track)
    rows to samples."""
    v = 10.0 if rng.random() < 0.3 else 12.0
    if rng.random() < 0.1:
        v *= C.SURGE_MULT
    R = C.FOG_RANGE if rng.random() < 0.1 else C.SENSOR_RANGE
    close = max(v + m.s, 0.5)
    t_first = float(rng.uniform(0.0, 1000.0))
    gap0 = R - float(rng.uniform(0.0, 1.0)) * close * C.DT
    g = T.GROUP[m.kind]
    tr = None
    k = 0
    while True:
        age = k * C.DT
        t = t_first + age
        gap = gap0 - close * age
        ta, tb = (gap if gap > 0 else 0.0) / close, (gap + m.w + C.HIT_W) / close
        if tb < 0.0:
            break
        y = m.y_at(t)
        tr = T.Track.start(t, y) if tr is None else tr.add(t, y)
        lo, hi = T.core_band(tr, t, ta, tb, g)
        truth = [m.y_at(t + ta + (tb - ta) * i / 12.0) for i in range(13)]
        samples.append((g, age, tb, lo - min(truth), max(truth) - hi, min(truth), max(truth), tr, ta))
        k += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--birds", type=int, default=20000)
    ap.add_argument("--traffic", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(C.TRACKER_YAML))
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    gen = Simulator("L1", 0, generate=False).gen
    t0 = time.time()
    samples = []
    for kind, n in ((C.BIRD, args.birds), (C.TRAFFIC, args.traffic)):
        got = 0
        while got < n:
            _st, movers = gen._sample(kind, 0.0, rng)
            for m in movers:
                sightings(m, rng, samples)
                got += 1
    print(f"{len(samples):,} sightings in {time.time() - t0:.0f} s")

    qs = C.TRACK_QUANTILES
    G = np.array([s[0] for s in samples])
    A = np.array([T._bin(T.AGE_EDGES, s[1]) for s in samples])
    H = np.array([T._bin(T.AHEAD_EDGES, s[2]) for s in samples])
    E = np.array([(s[3], s[4]) for s in samples])
    table = {}
    full = np.zeros((3, N_AGE, N_AHEAD, 2))
    for g, name in ((T.G_BIRD, "bird"), (T.G_TRAFFIC, "traffic")):
        m = np.full((N_AGE, N_AHEAD, 2), np.nan)
        for i in range(N_AGE):
            for j in range(N_AHEAD):
                sel = (G == g) & (A == i) & (H == j)
                if sel.sum() >= 30:
                    m[i, j] = np.maximum(np.quantile(E[sel], qs[j], axis=0), 0.02)
        # cells (almost) never visited, e.g. a just-seen bird passing within 0.4 s or an old track
        # with a long look ahead: the nearest visited cell of the same watched-time row (the
        # further-ahead one on a tie); a row without any: the younger row's; else the span
        for i in range(N_AGE):
            have = np.flatnonzero(~np.isnan(m[i, :, 0]))
            for j in range(N_AHEAD):
                if not np.isnan(m[i, j, 0]):
                    continue
                if have.size:
                    k = have[np.lexsort((-have, np.abs(have - j)))[0]]
                    m[i, j] = m[i, k]
                else:
                    m[i, j] = m[i - 1, j] if i > 0 else T.SPAN[g]
        for i in range(N_AGE - 2, -1, -1):     # younger >= older
            m[i] = np.maximum(m[i], m[i + 1])
        for j in range(1, N_AHEAD):            # further ahead >= nearer
            m[:, j] = np.maximum(m[:, j], m[:, j - 1])
        m = np.minimum(m, T.SPAN[g])
        full[g] = np.round(m, 3)
        table[name] = {"down": np.round(m[..., 0], 3).tolist(), "up": np.round(m[..., 1], 3).tolist()}

    out = {"note": "SPEC v2.3 tracker margins (m) by type, watched-time bin and ahead bin; written by "
                   "scripts/calibrate_tracker.py. Rows: watched < " + ", ".join(f"{e:g}" for e in T.AGE_EDGES) +
                   ", more (s). Columns: pass window ends < " + ", ".join(f"{e:g}" for e in T.AHEAD_EDGES) +
                   ", more (s) ahead.",
           "quantile_by_ahead": list(qs), "birds": args.birds, "traffic": args.traffic, "seed": args.seed, "margins": table}
    Path(args.out).write_text(yaml.safe_dump(out, sort_keys=False, width=200))
    T._MARGINS = full
    print(f"wrote {args.out}")

    # how the finished band does (same sightings, in-sample), by watched time and by look-ahead
    print("| type | sightings | band holds the mover | median band height (centre) |")
    print("|---|---|---|---|")
    for g, name in ((T.G_BIRD, "bird"), (T.G_TRAFFIC, "traffic")):
        groups = [(f"watched {a:g}-{b:g} s", lambda s, a=a, b=b: a <= s[1] < b)
                  for a, b in ((0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 9.0))]
        groups += [(f"pass ends {a:g}-{b:g} s ahead", lambda s, a=a, b=b: a <= s[2] < b)
                   for a, b in ((0.0, 0.4), (0.4, 0.8), (0.8, 1.2), (1.2, 9.0))]
        for label, keep in groups:
            rows = [s for s in samples if s[0] == g and keep(s)][::2]
            if not rows:
                continue
            ok, width = 0, []
            for (_g, age, tb, _ed, _eu, tlo, thi, tr, ta) in rows:
                lo, hi = T.finish(T.core_band(tr, tr.ts[-1], ta, tb, g), tr, age, tb, g)
                ok += (lo <= tlo + 1e-9) and (thi <= hi + 1e-9)
                width.append(hi - lo)
            print(f"| {name}, {label} | {len(rows):,} | {ok / len(rows):.1%} | {np.median(width):.2f} m |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
