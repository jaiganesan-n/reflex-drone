"""Dataset card: ``data/<name>/CARD.md``.

    uv run python -m data.inspect data/teacher
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

from data.dataset import label_entropy, list_shards, load_frames, read_meta
from drone_env import constants as C
from teacher.planner import soft_targets

HOLD_WARN = 0.85
THREAT_WARN = 0.10


def _table(rows, header) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def card(d: str | Path) -> tuple[str, list[str]]:
    d = Path(d)
    meta = read_meta(d) or {}
    a = load_frames(d)
    n = len(a["step"])
    T = float((meta.get("teacher") or {}).get("label_temperature", 2.0))
    p = soft_targets(a["teacher_scores"], T)
    ent = label_entropy(p)
    lines = [f"# Dataset card: `{d}`", ""]
    t = meta.get("teacher", {})
    lines += [f"- **Frames:** {n:,} in {len(list_shards(d))} shards, {len(np.unique(a['ep_key'])):,} episodes",
              f"- **Kind:** {meta.get('kind', '?')} · seeds: {(meta.get('collect') or {}).get('seeds', '?')} "
              f"({len(meta.get('seeds_used', [])):,} used) · spec {meta.get('spec_version', '?')}",
              f"- **Teacher:** beam {t.get('beam')}, depth {t.get('depth')} x {t.get('macro_frames')} frames, "
              f"hold_gust {t.get('hold_gust')}, jump_extra_penalty {t.get('jump_extra_penalty')}, "
              f"duck_extra_penalty {t.get('duck_extra_penalty')}, "
              f"label T {T}",
              f"- **Config hash:** `{meta.get('config_hash', '?')[:16]}` · settings `{meta.get('settings_hash', '?')}`",
              ""]
    # levels
    rows = []
    for code, name in enumerate(C.LEVELS):
        m = a["level"] == code
        if m.any():
            eps = len(np.unique(a["ep_key"][m]))
            rows.append((name, f"{int(m.sum()):,}", f"{100 * m.mean():.1f}%", eps, f"{m.sum() / max(eps, 1):.0f}"))
    lines += ["## Frames by level", "", _table(rows, ["level", "frames", "share", "episodes", "frames/episode"]), ""]
    # episode length and speed coverage (the speed ramps with distance)
    eps_len = np.bincount(np.unique(a["ep_key"], return_inverse=True)[1])
    speed = a["self"][:, 4].astype(np.float64) * 8.0 + 6.0
    rows = [(f"> {v} m/s", f"{100 * (speed > v).mean():.1f}%") for v in (7, 8, 10, 12)]
    lines += ["## Episodes and speed", "",
              f"- Frames per episode: mean {eps_len.mean():.0f}, median {np.median(eps_len):.0f}, "
              f"max {eps_len.max()} (cap {(meta.get('collect') or {}).get('frames_cap', '?')})",
              f"- Speed: mean {speed.mean():.1f} m/s, max {speed.max():.1f} m/s", "",
              _table(rows, ["speed", "share of frames"]), ""]
    # teacher actions
    cnt = np.bincount(a["action"].astype(np.int64), minlength=5)
    rows = [(C.ACTION_NAMES[i], f"{cnt[i]:,}", f"{100 * cnt[i] / n:.1f}%") for i in range(5)]
    lines += ["## Teacher action", "", _table(rows, ["action", "frames", "share"]), ""]
    # threat
    th = np.bincount(a["threat"].astype(np.int64), minlength=5)
    labels = ["0 (> 2 s)", "1 (1-2 s)", "2 (0.5-1 s)", "3 (< 0.5 s)", "4 (unavoidable)"]
    rows = [(labels[i], f"{th[i]:,}", f"{100 * th[i] / n:.1f}%") for i in range(5)]
    lines += ["## Threat", "", _table(rows, ["threat", "frames", "share"]), ""]
    # tags
    tags = a["tags"].sum(0)
    rows = [(nm, f"{int(tags[i]):,}", f"{100 * tags[i] / n:.1f}%") for i, nm in enumerate(("gust", "fog", "surge"))]
    rows.append(("any", f"{int(a['tags'].any(1).sum()):,}", f"{100 * a['tags'].any(1).mean():.1f}%"))
    lines += ["## Interruption tags", "", _table(rows, ["tag", "frames", "share"]), ""]
    # labels
    avail = a["action_mask"]
    marg = cnt / n
    marg_ent = -(marg[marg > 0] * np.log(marg[marg > 0])).sum()
    lines += ["## Labels", "",
              f"- Soft-label entropy (p_teacher, T={T}): mean {ent.mean():.3f} nats "
              f"(max for 5 actions {np.log(5):.3f}); median {np.median(ent):.3f}",
              f"- Teacher-action marginal entropy: {marg_ent:.3f} nats",
              f"- Frames with no safe action (threat 4): {100 * (a['threat'] == 4).mean():.1f}%",
              f"- DAgger frames (student_action >= 0): {int((a['student_action'] >= 0).sum()):,}", ""]
    rows = []
    for i in range(5):
        m = avail[:, i]
        rows.append((C.ACTION_NAMES[i], f"{100 * m.mean():.1f}%",
                     f"{100 * a['safe'][m, i].mean():.1f}%" if m.any() else "n/a"))
    lines += ["## Safe rate per action (frames where the action is available)", "",
              _table(rows, ["action", "available", "safe"]), ""]
    # warnings
    warns = []
    if marg[0] > HOLD_WARN:
        warns.append(f"HOLD is {100 * marg[0]:.1f}% of teacher actions (> {100 * HOLD_WARN:.0f}%). "
                     f"Raise burst_prob (e.g. 0.15) and top up.")
    thr2 = (a["threat"] >= 2).mean()
    if thr2 < THREAT_WARN:
        warns.append(f"threat >= 2 on only {100 * thr2:.1f}% of frames (< {100 * THREAT_WARN:.0f}%). "
                     f"Raise burst_prob (e.g. 0.15) and top up.")
    cap = (meta.get("collect") or {}).get("frames_cap")
    if meta.get("kind") == "teacher" and cap and eps_len.mean() < 0.3 * float(cap):
        warns.append(f"episodes average {eps_len.mean():.0f} frames (cap {cap}): crashes cut them short, so "
                     f"later, faster states are rare (only {100 * (speed > 8).mean():.1f}% of frames above 8 m/s). "
                     f"Lower burst_prob, or check the teacher (planner --check).")
    lines += ["## Warnings", ""] + ([f"- **WARNING:** {w}" for w in warns] or ["- none"]) + [""]
    return "\n".join(lines), warns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data.inspect", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="dataset dir, e.g. data/teacher")
    args = ap.parse_args(argv)
    text, warns = card(args.data)
    out = Path(args.data) / "CARD.md"
    out.write_text(text)
    print(text)
    print(f"\n-> {out}")
    for w in warns:
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
