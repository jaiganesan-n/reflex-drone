"""HUD band, buttons and banners for the DroneDash app (display only), plus the window.

Layout is designed on a WORLD_PX_W x 500 grid (world 1040 x 400 since v2.3, + HUD band 100) and drawn at
LAYOUT = 1.2 times that size, so the window is 1248 x 600 logical pixels (48 px per metre),
then multiplied by the Retina factor when the screen has one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from drone_env import constants as C
from drone_env.render import WORLD_PX_H, WORLD_PX_W, WorldPainter

from .controls import State

LAYOUT = 1.2                                    # window size factor over the design grid
RIGHT_X = WORLD_PX_W - 188                      # right-hand HUD column (buttons, hints)
HUD_H = 100                                     # design px
WIN_W = int(round(WORLD_PX_W * LAYOUT))         # 1248 logical px
WIN_H = int(round((WORLD_PX_H + HUD_H) * LAYOUT))   # 600 logical px

# light theme
BG = (248, 250, 252)
BAND_EDGE = (222, 228, 236)
PANEL = (236, 241, 246)
TEXT = (30, 41, 59)
DIM = (100, 116, 139)
FAINT = (190, 200, 212)
ACCENT = (249, 115, 22)
GOOD = (34, 160, 90)
WARN = (234, 162, 20)
BAD = (220, 60, 60)
ACTION_COLORS = [(148, 163, 184), (59, 130, 246), (16, 185, 129), (249, 115, 22), (168, 85, 247)]
THREAT_COLORS = [(74, 180, 96), (150, 200, 80), (240, 180, 40), (245, 124, 0), (226, 64, 54)]
SHORT = ["HOLD", "ASC", "DESC", "JUMP", "DUCK"]


@dataclass
class HudData:
    state: State
    agent: str
    level: str
    level_now: str = ""
    seed: int | None = None
    distance_m: float = 0.0
    speed: float = 0.0
    action: int | None = None
    decision: object | None = None          # app.agents_base.Decision
    action_mask: np.ndarray | None = None
    ms_last: float | None = None
    ms_p99: float | None = None
    summary: dict = field(default_factory=dict)
    seed_mode: str = "random"
    enabled: dict = field(default_factory=dict)


class Display:
    """Window with crisp Retina output when available, plus an offscreen canvas.

    Tries pygame.Window(allow_high_dpi) + Renderer (draws at the real pixel density), falls
    back to pygame.display.set_mode (scale 1). Headless (SDL_VIDEODRIVER=dummy) works too.
    """

    def __init__(self, title: str = "DroneDash", hidden: bool = False):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        self.pg = pygame
        if not pygame.display.get_init():
            pygame.display.init()
        pygame.font.init()
        self.window = self.renderer = self.texture = None
        self.scale = 1.0
        try:
            from pygame._sdl2.video import Renderer, Texture
            self.window = pygame.Window(title, (WIN_W, WIN_H), allow_high_dpi=True, hidden=hidden)
            self.renderer = Renderer(self.window)
            vp = self.renderer.get_viewport()
            self.scale = max(1.0, round(vp.w / WIN_W * 4) / 4)
            self.size_px = (int(WIN_W * self.scale), int(WIN_H * self.scale))
            self.texture = Texture(self.renderer, self.size_px, streaming=True)
        except Exception:
            if self.window is not None:
                try:
                    self.window.destroy()
                except Exception:
                    pass
            self.window = self.renderer = self.texture = None
            self.scale = 1.0
            self.size_px = (WIN_W, WIN_H)
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            pygame.display.set_caption(title)
        self.canvas = pygame.Surface(self.size_px)

    def present(self) -> None:
        if self.renderer is not None:
            self.texture.update(self.canvas)
            self.renderer.clear()
            self.renderer.blit(self.texture, self.pg.Rect(0, 0, *self.size_px))
            self.renderer.present()
        else:
            self.screen.blit(self.canvas, (0, 0))
            self.pg.display.flip()

    def close(self) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None
        self.pg.display.quit()


class Hud:
    """Draws on the canvas in design coordinates times ``s`` (= Retina factor x LAYOUT)."""

    def __init__(self, scale: float = 1.0, show_attention: bool = False):
        from drone_env.render import _pg
        pygame = _pg()                      # display + font initialised, no audio
        self.pg = pygame
        self.s = scale * LAYOUT
        self.show_attention = show_attention
        f = lambda px: pygame.font.Font(None, max(8, int(round(px * self.s))))   # noqa: E731
        self.f_big = f(34)
        self.f_mid = f(24)
        self.f_txt = f(19)
        self.f_small = f(16)
        self.f_tiny = f(14)
        self.f_title = f(72)
        self.f_banner = f(60)
        self.painter = WorldPainter(self.s)
        bx, by, bw, bh, gap = RIGHT_X, WORLD_PX_H + 14, 56, 28, 6
        # design rects (drawing) and logical-window rects (mouse hit tests)
        self.buttons_design = {name: pygame.Rect(bx + i * (bw + gap), by, bw, bh)
                               for i, name in enumerate(("start", "pause", "end"))}
        self.buttons = {name: pygame.Rect(round(r.x * LAYOUT), round(r.y * LAYOUT), round(r.w * LAYOUT),
                                          round(r.h * LAYOUT)) for name, r in self.buttons_design.items()}

    # -------------------------------------------------------------- helpers -----
    def R(self, x, y, w, h):
        s = self.s
        return self.pg.Rect(round(x * s), round(y * s), round(w * s), round(h * s))

    def text(self, surf, font, txt, color, x, y, anchor="topleft"):
        img = font.render(txt, True, color)
        r = img.get_rect(**{anchor: (round(x * self.s), round(y * self.s))})
        surf.blit(img, r)
        return r

    def button_at(self, pos) -> str | None:
        for name, r in self.buttons.items():
            if r.collidepoint(pos):
                return name
        return None

    # -------------------------------------------------------------- frame -------
    def draw(self, surf, sim, hud: HudData) -> None:
        attended_uid = None
        d = hud.decision
        if (self.show_attention and sim is not None and d is not None and d.attended is not None
                and 0 <= d.attended < len(sim.last_obs_uids)):
            attended_uid = sim.last_obs_uids[d.attended]
        if sim is not None:
            self.painter.draw(surf, sim, 0, 0, attended_uid=attended_uid)
        else:
            self.painter.draw_empty(surf, 0, 0)
        self.draw_band(surf, hud)
        if hud.state == State.IDLE:
            self.banner_idle(surf, hud)
        elif hud.state == State.PAUSED:
            self.banner_paused(surf)
        elif hud.state == State.OVER:
            self.banner_over(surf, hud)

    def draw_band(self, surf, hud: HudData) -> None:
        pg = self.pg
        surf.fill(BG, self.R(0, WORLD_PX_H, WORLD_PX_W, HUD_H))
        pg.draw.line(surf, BAND_EDGE, (0, round(WORLD_PX_H * self.s)), (round(WORLD_PX_W * self.s), round(WORLD_PX_H * self.s)),
                     max(1, round(self.s)))
        top = WORLD_PX_H
        # A: distance / speed / level / seed
        self.text(surf, self.f_tiny, "DISTANCE", DIM, 16, top + 10)
        dist = hud.distance_m
        self.text(surf, self.f_big, f"{dist / 1000:,.2f} km" if dist >= 1000 else f"{dist:,.1f} m", TEXT, 16, top + 24)
        self.text(surf, self.f_small, f"SPEED {hud.speed:4.1f} m/s", TEXT, 16, top + 56)
        lvl = hud.level if not hud.level_now or hud.level_now == hud.level else f"{hud.level} ({hud.level_now})"
        seed = "-" if hud.seed is None else str(hud.seed)
        self.text(surf, self.f_tiny, f"LEVEL {lvl}   SEED {seed}", DIM, 16, top + 76)
        # B: action + decision time
        x = 236
        self.text(surf, self.f_tiny, "ACTION", DIM, x, top + 10)
        if hud.action is not None:
            self.text(surf, self.f_mid, C.ACTION_NAMES[hud.action], ACTION_COLORS[hud.action], x, top + 26)
        else:
            self.text(surf, self.f_mid, "-", DIM, x, top + 26)
        self.text(surf, self.f_tiny, "DECISION TIME", DIM, x, top + 54)
        last = "-" if hud.ms_last is None else f"{hud.ms_last:.2f} ms"
        p99 = "-" if hud.ms_p99 is None else f"{hud.ms_p99:.2f} ms"
        self.text(surf, self.f_small, f"{last}  p99 {p99}", TEXT, x, top + 70)
        # C: intuition meter
        self.draw_meter(surf, hud, 420, top + 8)
        # D: threat chip + hesitation light
        x = 640
        self.text(surf, self.f_tiny, "THREAT", DIM, x, top + 10)
        thr = getattr(hud.decision, "threat", None) if hud.decision is not None else None
        chip = self.R(x, top + 26, 52, 24)
        col = THREAT_COLORS[int(thr)] if thr is not None else PANEL
        pg.draw.rect(surf, col, chip, border_radius=round(6 * self.s))
        self.text(surf, self.f_small, "-" if thr is None else str(int(thr)), (255, 255, 255) if thr is not None else DIM,
                  x + 26, top + 38, anchor="center")
        hes = bool(getattr(hud.decision, "hesitating", False)) if hud.decision is not None else False
        has_probs = hud.decision is not None and getattr(hud.decision, "probs", None) is not None
        cx, cy = x + 10, top + 72
        pg.draw.circle(surf, WARN if hes else FAINT, (round(cx * self.s), round(cy * self.s)), round(8 * self.s))
        label = "HESITATING" if hes else ("steady" if has_probs else "n/a")
        self.text(surf, self.f_tiny, label, WARN if hes else DIM, x + 24, cy, anchor="midleft")
        # E: buttons + hints
        for name, r in self.buttons_design.items():
            enabled = hud.enabled.get(name, False)
            label = "Resume" if (name == "pause" and hud.state == State.PAUSED) else name.capitalize()
            rr = self.R(r.x, r.y, r.w, r.h)
            primary = enabled and name == "start"
            pg.draw.rect(surf, ACCENT if primary else ((255, 255, 255) if enabled else PANEL), rr, border_radius=round(7 * self.s))
            pg.draw.rect(surf, ACCENT if primary else (FAINT if enabled else BAND_EDGE), rr, max(1, round(self.s)),
                         border_radius=round(7 * self.s))
            self.text(surf, self.f_small, label, (255, 255, 255) if primary else (TEXT if enabled else FAINT),
                      r.centerx, r.centery + 1, anchor="center")
        self.text(surf, self.f_tiny, "Enter Start · P Pause · Q End", DIM, RIGHT_X, top + 52)
        self.text(surf, self.f_tiny, f"agent: {hud.agent}", DIM, RIGHT_X, top + 72)

    def draw_meter(self, surf, hud: HudData, x0: int, y0: int) -> None:
        pg = self.pg
        d = hud.decision
        probs = getattr(d, "probs", None) if d is not None else None
        psafe = getattr(d, "p_safe", None) if d is not None else None
        mask = hud.action_mask
        self.text(surf, self.f_tiny, "INTUITION" + ("" if probs is not None else "  (no probabilities)"), DIM, x0, y0 + 2)
        bw, gap, bh = 30, 10, 50
        base = y0 + 18 + bh
        for a in range(5):
            bx = x0 + a * (bw + gap)
            outer = self.R(bx, base - bh, bw, bh)
            pg.draw.rect(surf, PANEL, outer, border_radius=round(4 * self.s))
            avail = mask is None or bool(mask[a])
            if probs is not None:
                h = float(np.clip(probs[a], 0, 1)) * bh
                if h > 0.5:
                    pg.draw.rect(surf, ACTION_COLORS[a], self.R(bx, base - h, bw, h), border_radius=round(4 * self.s))
            elif hud.action == a:
                pg.draw.rect(surf, ACTION_COLORS[a], self.R(bx, base - bh, bw, bh), border_radius=round(4 * self.s))
            if psafe is not None:
                ty = base - float(np.clip(psafe[a], 0, 1)) * bh
                pg.draw.line(surf, TEXT, (round((bx - 3) * self.s), round(ty * self.s)),
                             (round((bx + bw + 3) * self.s), round(ty * self.s)), max(2, round(2 * self.s)))
            if not avail:
                pg.draw.line(surf, FAINT, (outer.left + 4, outer.top + 4), (outer.right - 4, outer.bottom - 4), max(1, round(2 * self.s)))
                pg.draw.line(surf, FAINT, (outer.right - 4, outer.top + 4), (outer.left + 4, outer.bottom - 4), max(1, round(2 * self.s)))
            if hud.action == a:
                pg.draw.rect(surf, TEXT, outer, max(1, round(2 * self.s)), border_radius=round(4 * self.s))
            self.text(surf, self.f_tiny, SHORT[a], TEXT if avail else FAINT, bx + bw / 2, base + 9, anchor="center")

    # -------------------------------------------------------------- banners -----
    def _veil(self, surf, alpha=110):
        pg = self.pg
        v = pg.Surface(self.R(0, 0, WORLD_PX_W, WORLD_PX_H).size, pg.SRCALPHA)
        v.fill((255, 255, 255, alpha))
        surf.blit(v, (0, 0))

    def _panel(self, surf, w, h):
        pg = self.pg
        x, y = (WORLD_PX_W - w) / 2, (WORLD_PX_H - h) / 2
        r = self.R(x, y, w, h)
        for grow, a in ((10, 10), (6, 14), (3, 20)):
            g = round(grow * self.s)
            shadow = pg.Surface((r.w + 2 * g, r.h + 2 * g), pg.SRCALPHA)
            pg.draw.rect(shadow, (30, 41, 59, a), shadow.get_rect(), border_radius=round((16 + grow) * self.s))
            surf.blit(shadow, (r.x - g, r.y - g + round(3 * self.s)))
        panel = pg.Surface(r.size, pg.SRCALPHA)
        pg.draw.rect(panel, (255, 255, 255, 242), panel.get_rect(), border_radius=round(16 * self.s))
        pg.draw.rect(panel, (*BAND_EDGE, 255), panel.get_rect(), max(1, round(self.s)), border_radius=round(16 * self.s))
        surf.blit(panel, r.topleft)
        return x, y

    def banner_idle(self, surf, hud: HudData) -> None:
        self._veil(surf, 60)
        x, y = self._panel(surf, 560, 250)
        cx = WORLD_PX_W / 2
        self.text(surf, self.f_title, "DroneDash", ACCENT, cx, y + 58, anchor="center")
        self.text(surf, self.f_txt, "Reflex Drone · Project J'AI 2", DIM, cx, y + 100, anchor="center")
        self.text(surf, self.f_txt, f"agent: {hud.agent}   level: {hud.level}   seed: {hud.seed_mode}", TEXT, cx, y + 136,
                  anchor="center")
        self.text(surf, self.f_mid, "Press Enter (or click Start) to begin", TEXT, cx, y + 180, anchor="center")
        self.text(surf, self.f_small, "The drone flies itself. You can only Start, Pause and End a run.", DIM, cx, y + 216,
                  anchor="center")

    def banner_paused(self, surf) -> None:
        self._veil(surf, 120)
        cx, cy = WORLD_PX_W / 2, WORLD_PX_H / 2
        self.text(surf, self.f_banner, "PAUSED", TEXT, cx, cy - 14, anchor="center")
        self.text(surf, self.f_txt, "P / Space resume  ·  Q / Esc end", DIM, cx, cy + 30, anchor="center")

    def banner_over(self, surf, hud: HudData) -> None:
        s = hud.summary
        self._veil(surf, 80)
        x, y = self._panel(surf, 600, 300)
        cx = WORLD_PX_W / 2
        by = s.get("ended_by")
        if by == "crash":
            title, col = f"CRASHED · {s.get('crash_cause') or '?'}", BAD
        elif by == "user":
            title, col = "RUN ENDED", TEXT
        else:
            title, col = "RUN OVER", TEXT
        self.text(surf, self.f_banner, title, col, cx, y + 44, anchor="center")
        dist = s.get("distance_m", 0)
        rows = [
            ("Distance", f"{dist / 1000:,.2f} km" if dist >= 1000 else f"{dist:,.1f} m"),
            ("Time flown", f"{s.get('seconds_flown', 0):.1f} s"),
            ("Crash cause", s.get("crash_cause") or "none"),
            ("Seed", str(s.get("seed", "-"))),
            ("Decisions", f"{s.get('decisions', 0):,}"),
            ("Decision p50 / p99", f"{s.get('decision_ms_p50', 0):.3f} / {s.get('decision_ms_p99', 0):.3f} ms"),
            ("Hesitation share", f"{100 * s.get('hesitation_share', 0):.1f} %"),
            ("Overruns > 33 ms", str(s.get("overruns", 0))),
        ]
        for i, (k, v) in enumerate(rows):
            col_x = x + 40 + (i % 2) * 280
            row_y = y + 92 + (i // 2) * 38
            self.text(surf, self.f_tiny, k.upper(), DIM, col_x, row_y)
            self.text(surf, self.f_txt, v, TEXT, col_x, row_y + 14)
        self.text(surf, self.f_small, "Enter / S new run  ·  Q / Esc title", DIM, cx, y + 278, anchor="center")
