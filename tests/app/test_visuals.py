import numpy as np
import pygame

from app.agents_base import Decision
from app.controls import State
from app.hud import WIN_H, WIN_W, Hud, HudData
from drone_env import constants as C
from drone_env.render import ATTENDED
from drone_env.sim import Simulator


def test_window_is_20_percent_larger():
    assert (WIN_W, WIN_H) == (1248, 600)            # SPEC v2.3: 26 m world


def _frame(show_attention: bool):
    s = Simulator("L1", 0, generate=False)
    s.place_static(C.BUILDING, s.x + 4.0, s.x + 6.0, 0.0, 3.0)
    s.observe()
    hud = Hud(1.0, show_attention=show_attention)
    canvas = pygame.Surface((WIN_W, WIN_H))
    hud.draw(canvas, s, HudData(state=State.RUNNING, agent="t", level="L1", decision=Decision(0, attended=0)))
    px = pygame.surfarray.array3d(canvas)
    return int(np.all(px == np.array(ATTENDED), axis=2).sum())


def test_attention_outline_is_off_by_default():
    assert _frame(show_attention=False) == 0
    assert _frame(show_attention=True) > 0


def test_hud_shows_l5_plus_past_the_ramp(tmp_path):
    from app.play import App
    from drone_env import constants as C
    app = App("rule", "endless", runs_dir=tmp_path, hidden=True, quiet=True)
    try:
        app.info = {"distance_m": C.ENDLESS_RAMP_M - 1.0, "level_now": "L4"}
        assert app._level_label() == "L4"
        app.info = {"distance_m": C.ENDLESS_RAMP_M + 500.0, "level_now": "L5"}
        assert app._level_label() == "L5+"
    finally:
        app.close()


def test_planes_are_drawn_inside_their_hitbox():
    import pygame
    from drone_env import events
    from drone_env.render import WorldPainter
    s = Simulator("L5", 0, generate=False)
    events.make_calm(s)
    for sub, (w, h) in ((0, (5.0, 1.0)), (1, (3.3, 0.8))):
        s.movers = []
        m = s.place_mover(C.PLANE, s.x + 8.0, 6.0, w, h, s=6.0, sub=sub)
        wp = WorldPainter(1.0)
        surf = pygame.Surface((wp.w, wp.h))
        surf.fill((0, 0, 0))
        wp._plane(surf, pygame.Rect(100, 100, int(w * wp.ppm), int(h * wp.ppm)), m, 0.0)
        px = pygame.surfarray.array3d(surf).sum(axis=2)
        ys, xs = np.nonzero(px.T)
        assert xs.min() >= 100 - 2 and xs.max() <= 100 + w * wp.ppm + 2
        assert ys.min() >= 100 - 2 and ys.max() <= 100 + h * wp.ppm + 2
