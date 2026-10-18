import json

import numpy as np
import pygame
import pytest

from app.agents_base import Decision, RuleAgent, register_agent
from app.controls import Command, Controller, Effect, State, command_for_key

S, P, E, Q = Command.START, Command.PAUSE, Command.END, Command.QUIT


# ------------------------------------------------------------------ pure state machine --
@pytest.mark.parametrize("start,cmd,end,effects", [
    (State.IDLE, S, State.RUNNING, [Effect.NEW_RUN]),
    (State.IDLE, P, State.IDLE, []),
    (State.IDLE, E, State.IDLE, []),
    (State.RUNNING, P, State.PAUSED, [Effect.PAUSED]),
    (State.RUNNING, E, State.OVER, [Effect.RUN_OVER]),
    (State.RUNNING, S, State.RUNNING, []),
    (State.PAUSED, P, State.RUNNING, [Effect.RESUMED]),
    (State.PAUSED, E, State.OVER, [Effect.RUN_OVER]),
    (State.PAUSED, S, State.PAUSED, []),
    (State.OVER, S, State.RUNNING, [Effect.NEW_RUN]),
    (State.OVER, E, State.IDLE, [Effect.TO_TITLE]),
    (State.OVER, P, State.OVER, []),
])
def test_transitions(start, cmd, end, effects):
    c = Controller()
    c.state = start
    assert c.handle(cmd) == effects
    assert c.state == end


def test_ended_by_and_crash():
    c = Controller()
    c.handle(S)
    assert c.crashed() == [Effect.RUN_OVER] and c.state == State.OVER and c.ended_by == "crash"
    assert c.crashed() == []                         # crash outside RUNNING does nothing
    c.handle(S)
    c.handle(E)
    assert c.ended_by == "user"
    c.handle(S)
    c.handle(P)
    c.handle(E)
    assert c.ended_by == "user" and c.state == State.OVER


@pytest.mark.parametrize("state,effects", [
    (State.IDLE, [Effect.QUIT]), (State.OVER, [Effect.QUIT]),
    (State.RUNNING, [Effect.RUN_OVER, Effect.QUIT]), (State.PAUSED, [Effect.RUN_OVER, Effect.QUIT]),
])
def test_window_close_quits_from_any_state(state, effects):
    c = Controller()
    c.state = state
    assert c.handle(Q) == effects


def test_keys_and_buttons():
    for k in ("return", "s", "S"):
        assert command_for_key(k) == S
    for k in ("p", "space"):
        assert command_for_key(k) == P
    for k in ("q", "escape"):
        assert command_for_key(k) == E
    for k in ("up", "down", "w", "a", "d", "j", "k", "left", "right"):
        assert command_for_key(k) is None              # no flying controls, ever
    c = Controller()
    assert [c.button_enabled(b) for b in ("start", "pause", "end")] == [True, False, False]
    c.handle(S)
    assert [c.button_enabled(b) for b in ("start", "pause", "end")] == [False, True, True]
    c.handle(P)
    assert [c.button_enabled(b) for b in ("start", "pause", "end")] == [False, True, True]
    c.handle(E)
    assert [c.button_enabled(b) for b in ("start", "pause", "end")] == [True, False, True]


def test_should_act_only_while_running():
    c = Controller()
    seen = {c.state: c.should_act}
    for cmd in (S, P, P, E):
        c.handle(cmd)
        seen[c.state] = c.should_act
    assert seen == {State.IDLE: False, State.RUNNING: True, State.PAUSED: False, State.OVER: False}


# ------------------------------------------------------------------ headless app ---------
class CountingAgent(RuleAgent):
    def __init__(self):
        super().__init__()
        self.name = "counting"
        self.calls = 0
        self.resets = []

    def reset(self, action_desc):
        super().reset(action_desc)
        self.resets.append(np.array(action_desc))

    def act(self, obs):
        assert set(obs) == {"self", "rays", "hazards", "hazard_type", "hazard_mask", "action_mask"}
        self.calls += 1
        d = super().act(obs)
        return Decision(d.action, probs=np.full(5, 0.2), hesitating=self.calls % 2 == 0)


class DiveAgent(CountingAgent):
    def act(self, obs):
        self.calls += 1
        return Decision(2)


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0)


def click(app, name):
    r = app.hud.buttons[name]
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=r.center, button=1)


@pytest.fixture
def make_app(tmp_path):
    from app.play import App
    apps = []

    def _make(agent="counting", **kw):
        holder = {}
        register_agent("counting", lambda arg: holder.setdefault("a", CountingAgent()))
        register_agent("dive", lambda arg: holder.setdefault("a", DiveAgent()))
        app = App(agent, kw.pop("level", "L1"), runs_dir=tmp_path, hidden=True, quiet=True, **kw)
        apps.append(app)
        return app, holder["a"]
    yield _make
    for a in apps:
        a.close()


def test_app_start_pause_resume_end_and_no_act_while_paused_or_over(make_app, tmp_path):
    app, agent = make_app()
    for _ in range(5):
        app.tick([])
    assert app.ctrl.state == State.IDLE and agent.calls == 0           # no act() at the title
    app.tick([key(pygame.K_RETURN)])
    assert app.ctrl.state == State.RUNNING and len(agent.resets) == 1
    assert agent.resets[0].shape == (5, 3)
    for _ in range(20):
        app.tick([])
    n = agent.calls
    assert n >= 20 and app.env.unwrapped.sim.frame == n                 # lockstep: 1 act per frame
    app.tick([key(pygame.K_p)])
    frame = app.env.unwrapped.sim.frame
    for _ in range(30):
        app.tick([])
    assert app.ctrl.state == State.PAUSED and agent.calls == n          # frozen
    assert app.env.unwrapped.sim.frame == frame
    app.tick([key(pygame.K_SPACE)])
    for _ in range(10):
        app.tick([])
    assert agent.calls > n and app.ctrl.state == State.RUNNING
    app.tick([key(pygame.K_ESCAPE)])
    assert app.ctrl.state == State.OVER and app.ctrl.ended_by == "user"
    m = agent.calls
    for _ in range(10):
        app.tick([])
    assert agent.calls == m                                             # no act() while OVER
    log = json.loads(app.last_log_path.read_text())
    for k in ("seed", "level", "agent", "distance_m", "frames", "seconds_flown", "crash_cause", "ended_by",
              "decisions", "action_counts", "decision_ms_p50", "decision_ms_p99", "overruns",
              "hesitation_share"):
        assert k in log
    assert log["ended_by"] == "user" and log["decisions"] == m == log["frames"]
    assert log["seconds_flown"] == pytest.approx(log["frames"] / 30, abs=1e-3)   # paused time excluded
    assert 0.4 < log["hesitation_share"] < 0.6
    assert app.last_log_path.parent == tmp_path and app.last_log_path.name.endswith(f"_seed{log['seed']}.json")
    app.tick([key(pygame.K_q)])
    assert app.ctrl.state == State.IDLE


def test_buttons_drive_the_app(make_app):
    app, agent = make_app()
    app.tick([click(app, "pause")])                                     # disabled at the title
    assert app.ctrl.state == State.IDLE
    app.tick([click(app, "start")])
    assert app.ctrl.state == State.RUNNING
    app.tick([click(app, "pause")])
    assert app.ctrl.state == State.PAUSED
    app.tick([click(app, "pause")])
    assert app.ctrl.state == State.RUNNING
    app.tick([click(app, "end")])
    assert app.ctrl.state == State.OVER
    app.tick([click(app, "end")])
    assert app.ctrl.state == State.IDLE


def test_fresh_seed_each_start_unless_fixed(make_app):
    app, _ = make_app()
    seeds = []
    for _ in range(3):
        app.tick([key(pygame.K_s)])
        app.tick([])
        seeds.append(app.seed)
        app.tick([key(pygame.K_q)])
    assert len(set(seeds)) == 3
    app2, _ = make_app(seed=77)
    s2 = []
    for _ in range(2):
        app2.tick([key(pygame.K_s)])
        s2.append(app2.seed)
        app2.tick([key(pygame.K_q)])
    assert s2 == [77, 77]


def test_crash_ends_the_run_and_logs_frames(make_app):
    app, agent = make_app("dive", log_frames=True)
    app.tick([key(pygame.K_RETURN)])
    for _ in range(300):
        app.tick([])
        if app.ctrl.state == State.OVER:
            break
    assert app.ctrl.state == State.OVER and app.ctrl.ended_by == "crash"
    log = json.loads(app.last_log_path.read_text())
    assert log["crash_cause"] == "ground" and log["ended_by"] == "crash"
    csv_path = app.last_log_path.with_suffix(".csv")
    rows = csv_path.read_text().strip().splitlines()
    assert rows[0].startswith("frame,action,p_hold") and len(rows) == log["decisions"] + 1
    n = agent.calls
    for _ in range(5):
        app.tick([])
    assert agent.calls == n


def test_window_close_writes_log_and_quits(make_app):
    app, _ = make_app()
    app.tick([key(pygame.K_RETURN)])
    app.tick([])
    alive = app.tick([pygame.event.Event(pygame.QUIT)])
    assert alive is False and app.running is False
    assert json.loads(app.last_log_path.read_text())["ended_by"] == "quit"
