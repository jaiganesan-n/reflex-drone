"""DroneDash desktop app (SPEC §9). The drone flies itself; a person can only Start, Pause and End.

    uv run python -m app.play --agent rule
    uv run python -m app.play --agent onnx-arc:artifacts/arc.onnx --level endless --seed 42 --log-frames

The agent, level and seed are fixed at launch. Every Start uses a fresh random seed unless
--seed was given. Lockstep at 30 FPS: one agent.act() and one simulation step per rendered frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np

import gymnasium as gym

import drone_env  # noqa: F401  (registers DroneDash-v0)
from drone_env import constants as C
from drone_env.seeds import play_seed

from .agents_base import Decision, load_agent
from .controls import Command, Controller, Effect, State, command_for_button, command_for_key

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs" / "play"
OVERRUN_MS = 1000.0 / C.FPS          # 33.3 ms


class App:
    def __init__(self, agent_spec: str = "rule", level: str = "endless", seed: int | None = None,
                 log_frames: bool = False, runs_dir: Path | None = None, hidden: bool = False,
                 quiet: bool = False, show_attention: bool = False):
        if level not in C.LEVELS:
            raise ValueError(f"unknown level {level!r}; choose one of {C.LEVELS}")
        self.agent_spec = agent_spec
        self.agent = load_agent(agent_spec)
        self.level = level
        self.fixed_seed = seed
        self.log_frames = log_frames
        self.runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
        self.quiet = quiet
        self.ctrl = Controller()
        from .hud import Display, Hud
        self.display = Display(title=f"DroneDash · {agent_spec} · {level}", hidden=hidden)
        self.hud = Hud(self.display.scale, show_attention=show_attention)
        self.env = None
        self.obs = None
        self.info: dict = {}
        self.decision: Decision | None = None
        self.running = True
        self.summary: dict = {}
        self.last_log_path: Path | None = None
        self.act_calls = 0               # total act() calls over the app's life (tests use it)
        self._reset_stats()

    # ------------------------------------------------------------------ runs ---------
    def _reset_stats(self) -> None:
        self.seed = None
        self.ms: list[float] = []
        self.rolling: deque = deque(maxlen=300)
        self.actions: Counter = Counter()
        self.hesitations = 0
        self.overruns = 0
        self.frame_rows: list = []
        self.started_at = None
        self.run_wall = 0.0

    def new_run(self) -> None:
        if self.env is not None:
            self.env.close()
        self._reset_stats()
        self.seed = self.fixed_seed if self.fixed_seed is not None else play_seed()
        self.env = gym.make("DroneDash-v0", level=self.level, render_mode=None, max_frames=None)
        self.env.unwrapped.sensor_spec = getattr(self.agent, "sensor_spec", C.SENSOR_SPEC)
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.agent.reset(self.env.unwrapped.action_descriptors())
        self.decision = None
        self.started_at = datetime.now()
        self.summary = {}
        if not self.quiet:
            print(f"[run] start  agent={self.agent_spec} level={self.level} seed={self.seed}", flush=True)

    def step_once(self) -> None:
        """One lockstep frame: one agent decision, one simulation step."""
        t0 = time.perf_counter()
        d = self.agent.act(self.obs)
        ms = (time.perf_counter() - t0) * 1000.0
        self.act_calls += 1
        self.decision = d
        a = int(d.action)
        self.ms.append(ms)
        self.rolling.append(ms)
        if ms > OVERRUN_MS:
            self.overruns += 1
            if not self.quiet:
                print(f"[overrun] frame {self.info.get('frame', 0)}: act() took {ms:.1f} ms", flush=True)
        if d.hesitating:
            self.hesitations += 1
        if self.log_frames:
            p = d.probs if d.probs is not None else [None] * 5
            self.frame_rows.append([self.info.get("frame", 0), a, *[None if v is None else float(v) for v in p],
                                    round(ms, 4), round(self.info.get("distance_m", 0.0), 3),
                                    round(float(self.obs["self"][0]) * 10, 3), d.threat, int(bool(d.hesitating))])
        self.obs, _, terminated, truncated, self.info = self.env.step(a)
        self.actions[C.ACTION_NAMES[self.info["action"]]] += 1
        if terminated or truncated:
            self._apply(self.ctrl.crashed() if terminated else self.ctrl.handle(Command.END))

    def finalize_run(self) -> dict:
        ms = np.array(self.ms) if self.ms else np.zeros(1)
        info = self.info
        frames = int(info.get("frame", 0))
        s = {
            "seed": self.seed,
            "level": self.level,
            "agent": self.agent_spec,
            "distance_m": round(float(info.get("distance_m", 0.0)), 3),
            "frames": frames,
            "seconds_flown": round(frames * C.DT, 3),
            "crash_cause": info.get("crash_cause"),
            "ended_by": self.ctrl.ended_by,
            "decisions": len(self.ms),
            "action_counts": {n: self.actions.get(n, 0) for n in C.ACTION_NAMES},
            "decision_ms_p50": round(float(np.percentile(ms, 50)), 4),
            "decision_ms_p99": round(float(np.percentile(ms, 99)), 4),
            "overruns": self.overruns,
            "hesitation_share": round(self.hesitations / max(len(self.ms), 1), 4),
            # extras
            "level_reached": info.get("level_now", self.level),
            "masked_action_used": info.get("masked_action_used", 0),
            "obstacles_cleared": info.get("cleared", 0),
            "wall_seconds_running": round(self.run_wall, 3),
            "started_at": self.started_at.isoformat(timespec="seconds") if self.started_at else None,
            "spec_version": C.SPEC_VERSION,
        }
        self.summary = s
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = (self.started_at or datetime.now()).strftime("%Y%m%d-%H%M%S")
        stem = f"{stamp}_seed{self.seed}"
        path = self.runs_dir / f"{stem}.json"
        n = 2
        while path.exists():
            path = self.runs_dir / f"{stem}_{n}.json"
            n += 1
        path.write_text(json.dumps(s, indent=2))
        if self.log_frames:
            with open(path.with_suffix(".csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "action", "p_hold", "p_ascend", "p_descend", "p_jump", "p_duck",
                            "decision_ms", "distance_m", "y", "threat", "hesitating"])
                w.writerows(self.frame_rows)
        self.last_log_path = path
        if not self.quiet:
            print(f"[run] over   ended_by={s['ended_by']} distance={s['distance_m']:.1f} m "
                  f"cause={s['crash_cause']} p99={s['decision_ms_p99']:.3f} ms -> {path}", flush=True)
        return s

    # ------------------------------------------------------------------ events -------
    def _apply(self, effects) -> None:
        for e in effects:
            if e == Effect.NEW_RUN:
                self.new_run()
            elif e == Effect.RUN_OVER:
                self.finalize_run()
            elif e == Effect.QUIT:
                self.running = False

    def command_from_event(self, ev) -> Command | None:
        pg = self.display.pg
        if ev.type == pg.QUIT or ev.type == getattr(pg, "WINDOWCLOSE", -1):
            return Command.QUIT
        if ev.type == pg.KEYDOWN:
            return command_for_key(pg.key.name(ev.key))
        if ev.type == pg.MOUSEBUTTONDOWN and getattr(ev, "button", 1) == 1:
            name = self.hud.button_at(ev.pos)
            if name is not None and self.ctrl.button_enabled(name):
                return command_for_button(name)
        return None

    def tick(self, events=()) -> bool:
        """Process input, advance one frame if RUNNING, render. Returns False once quit."""
        for ev in events:
            cmd = self.command_from_event(ev)
            if cmd is not None:
                self._apply(self.ctrl.handle(cmd))
                if not self.running:
                    return False
        if self.ctrl.should_act:
            t0 = time.perf_counter()
            self.step_once()
            self.run_wall += time.perf_counter() - t0
        self.render()
        return self.running

    # ------------------------------------------------------------------ drawing ------
    def hud_data(self):
        from .hud import HudData
        sim = self.env.unwrapped.sim if self.env is not None else None
        r = np.array(self.rolling) if self.rolling else None
        return HudData(
            state=self.ctrl.state, agent=self.agent_spec, level=self.level,
            level_now=self._level_label(),
            seed=self.seed, distance_m=float(self.info.get("distance_m", 0.0)) if self.info else 0.0,
            speed=float(self.info.get("speed", 0.0)) if self.info else 0.0,
            action=self.info.get("action") if (self.info and self.decision is not None) else None,
            decision=self.decision,
            action_mask=self.obs["action_mask"] if self.obs is not None else None,
            ms_last=self.ms[-1] if self.ms else None,
            ms_p99=float(np.percentile(r, 99)) if r is not None else None,
            summary=self.summary, seed_mode="random" if self.fixed_seed is None else str(self.fixed_seed),
            enabled={b: self.ctrl.button_enabled(b) for b in ("start", "pause", "end")},
        ), sim

    def _level_label(self) -> str:
        """Level shown in the HUD: the active level, or "L5+" once endless is past the L5 ramp,
        where the speed stays at L5's and the obstacles keep getting denser (SPEC v1.7)."""
        if not self.info:
            return ""
        if self.level == "endless" and float(self.info.get("distance_m", 0.0)) >= C.ENDLESS_RAMP_M:
            return "L5+"
        return self.info.get("level_now", "")

    def render(self) -> None:
        hud, sim = self.hud_data()
        self.hud.draw(self.display.canvas, sim if self.ctrl.state != State.IDLE else None, hud)
        self.display.present()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.display.close()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.play", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="rule", help="agent spec: rule, random, onnx-arc:<path>, onnx-mlp:<path>")
    ap.add_argument("--level", default="endless", choices=C.LEVELS)
    ap.add_argument("--seed", type=int, default=None, help="fixed seed for every run (default: fresh random seed per Start)")
    ap.add_argument("--log-frames", action="store_true", help="also write a per-frame CSV next to the run log")
    ap.add_argument("--show-attention", action="store_true",
                    help="outline the hazard the agent attends to (Decision.attended); off by default")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        app = App(args.agent, args.level, args.seed, args.log_frames, show_attention=args.show_attention)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    pg = app.display.pg
    clock = pg.time.Clock()
    print("DroneDash: Enter/S = Start · P/Space = Pause · Q/Esc = End · close the window to quit", flush=True)
    try:
        while app.tick(pg.event.get()):
            clock.tick(C.FPS)
    except KeyboardInterrupt:
        if app.ctrl.state in (State.RUNNING, State.PAUSED):
            app._apply(app.ctrl.handle(Command.QUIT))
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.exit(main())
