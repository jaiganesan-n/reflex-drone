"""Start / Pause / End state machine (SPEC §9). Pure logic, no pygame.

A person can only Start, Pause/Resume and End a run. There are no flying controls.

    IDLE     Enter / S / Start button  -> RUNNING (new run)
    RUNNING  P / Space / Pause button  -> PAUSED
             Q / Esc / End button      -> OVER   (ended_by = "user")
             crash                     -> OVER   (ended_by = "crash")
    PAUSED   P / Space / Pause button  -> RUNNING
             Q / Esc / End button      -> OVER   (ended_by = "user")
    OVER     Enter / S / Start button  -> RUNNING (new run)
             Q / Esc / End button      -> IDLE
    any      window closed             -> quit
"""
from __future__ import annotations

from enum import Enum


class State(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    OVER = "OVER"


class Command(str, Enum):
    START = "start"
    PAUSE = "pause"      # toggles pause / resume
    END = "end"
    QUIT = "quit"        # window closed


class Effect(str, Enum):
    NEW_RUN = "new_run"          # make a fresh env + agent.reset()
    PAUSED = "paused"
    RESUMED = "resumed"
    RUN_OVER = "run_over"        # finalize the run log (ended_by in Controller.ended_by)
    TO_TITLE = "to_title"
    QUIT = "quit"


# key names as returned by pygame.key.name()
KEYMAP = {
    "return": Command.START, "enter": Command.START, "s": Command.START,
    "p": Command.PAUSE, "space": Command.PAUSE,
    "q": Command.END, "escape": Command.END,
}

BUTTONS = ("start", "pause", "end")
_BUTTON_CMD = {"start": Command.START, "pause": Command.PAUSE, "end": Command.END}


def command_for_key(key_name: str) -> Command | None:
    return KEYMAP.get(key_name.lower())


def command_for_button(button: str) -> Command:
    return _BUTTON_CMD[button]


class Controller:
    def __init__(self):
        self.state = State.IDLE
        self.ended_by: str | None = None
        self.runs_started = 0

    @property
    def should_act(self) -> bool:
        """The agent decides and the simulation steps only while RUNNING."""
        return self.state == State.RUNNING

    def button_enabled(self, button: str) -> bool:
        s = self.state
        if button == "start":
            return s in (State.IDLE, State.OVER)
        if button == "pause":
            return s in (State.RUNNING, State.PAUSED)
        if button == "end":
            return s in (State.RUNNING, State.PAUSED, State.OVER)
        return False

    def handle(self, cmd: Command) -> list[Effect]:
        s = self.state
        if cmd == Command.QUIT:
            out = []
            if s in (State.RUNNING, State.PAUSED):
                self.ended_by = "quit"
                out.append(Effect.RUN_OVER)
            self.state = State.OVER if out else s
            out.append(Effect.QUIT)
            return out
        if s == State.IDLE:
            if cmd == Command.START:
                return self._start()
        elif s == State.RUNNING:
            if cmd == Command.PAUSE:
                self.state = State.PAUSED
                return [Effect.PAUSED]
            if cmd == Command.END:
                return self._over("user")
        elif s == State.PAUSED:
            if cmd == Command.PAUSE:
                self.state = State.RUNNING
                return [Effect.RESUMED]
            if cmd == Command.END:
                return self._over("user")
        elif s == State.OVER:
            if cmd == Command.START:
                return self._start()
            if cmd == Command.END:
                self.state = State.IDLE
                return [Effect.TO_TITLE]
        return []

    def crashed(self) -> list[Effect]:
        if self.state == State.RUNNING:
            return self._over("crash")
        return []

    def _start(self) -> list[Effect]:
        self.state = State.RUNNING
        self.ended_by = None
        self.runs_started += 1
        return [Effect.NEW_RUN]

    def _over(self, by: str) -> list[Effect]:
        self.state = State.OVER
        self.ended_by = by
        return [Effect.RUN_OVER]
