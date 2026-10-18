"""Realistic, light-coloured world renderer (viewer only: agents never see any of this).

Design size is 40 px per metre: the 26 m x 10 m world (SPEC v2.3; 24 m before) is 1040 x 400 design px. ``scale``
multiplies everything (the app uses 1.2 x the Retina factor). Every drawn shape follows the
collision shape it stands for; decoration (clouds, far hills, rain, rotor discs) is subtle.

Static hazards are pre-rendered once into cached sprites keyed by their geometry, so a frame
is mostly blits and stays far below the 33 ms frame budget even on a Retina window.
"""
from __future__ import annotations

import math
import os
import random

import numpy as np

from . import constants as C

PPM = 40.0                             # design px per metre
WORLD_PX_W = int(round(PPM * C.WORLD_W))
WORLD_PX_H = int(round(PPM * C.WORLD_H))

# light, natural palette ------------------------------------------------------------------
SKY_TOP = (140, 192, 234)
SKY_BOTTOM = (226, 240, 250)
SKY = SKY_BOTTOM                        # flat fallback colour
SUN = (255, 246, 214)
GRASS_TOP = (132, 186, 82)
GRASS_BOTTOM = (100, 152, 58)
SOIL = (128, 96, 70)
GROUND = SOIL
HILL_FAR = (196, 218, 222)
HILL_NEAR = (176, 204, 196)
DRONE_BODY_TOP = (255, 172, 72)
DRONE_BODY_BOTTOM = (232, 112, 26)
DRONE_DARK = (54, 62, 72)
ROTOR_BLUR = (90, 98, 110)
BIRD_COLORS = ((72, 74, 80), (98, 84, 70), (58, 64, 74))
TRAFFIC_BODY = (214, 219, 225)
TRAFFIC_DARK = (78, 86, 98)
BALLOONS = (((226, 82, 64), (246, 200, 74)), ((58, 140, 214), (240, 244, 246)),
            ((60, 176, 110), (238, 142, 52)), ((142, 96, 186), (246, 210, 90)))
BASKET = (150, 104, 62)
# planes: body, belly, stripe, tail (liveries are picked per plane)
PLANE_LIVERIES = (((248, 249, 251), (214, 220, 228), (40, 96, 176), (40, 96, 176)),
                  ((250, 250, 250), (220, 222, 226), (206, 52, 52), (206, 52, 52)),
                  ((246, 248, 250), (210, 216, 224), (34, 150, 140), (250, 186, 40)),
                  ((244, 246, 248), (212, 216, 222), (90, 96, 108), (234, 104, 40)))
PLANE_WINDOW = (64, 84, 108)
STORM_TOP = (184, 190, 202)
STORM_BOTTOM = (122, 130, 146)
RAIN = (120, 150, 190)
DOWNDRAFT = (170, 210, 236)
FOG = (246, 248, 250)
ATTENDED = (255, 196, 0)
WIND = (64, 150, 220)
CHIP_BG = (255, 255, 255)
CHIP_FG = (40, 52, 66)
CHIP_EDGE = (200, 210, 222)
# v2.0 real-life obstacles
LEAVES = (((52, 112, 60), (84, 150, 74), (138, 192, 104)), ((46, 100, 72), (74, 138, 92), (124, 182, 120)),
          ((70, 116, 48), (104, 156, 64), (160, 200, 96)))     # dark, mid, light per tree
BARK = (112, 82, 58)
POLE_METAL = (128, 134, 142)
LAMP_GLOW = (255, 238, 176)
WIRE_DARK = (38, 42, 50)
BACK_POLE = (176, 160, 140)                # power-line poles and crane towers stand beside the road
CRANE_YELLOW = (246, 190, 40)
CRANE_DARK = (170, 122, 20)
KITE_COLORS = (((226, 64, 58), (250, 204, 64)), ((52, 120, 214), (246, 246, 246)),
               ((40, 168, 120), (250, 140, 50)), ((150, 82, 200), (255, 214, 90)))
# sprite margins (m) left / right of a parts-built obstacle's blocking span, for drawn decoration
PART_MARGINS = {C.TREE: (0.5, 0.5), C.POLE: (0.3, 0.3), C.POWERLINE: (0.4, 0.4), C.KITE: (0.4, 0.4),
                C.CRANE: (0.3, 4.4)}

BUILDING_STYLES = {
    # facade top, facade bottom, window, window frame/mullion, lit window
    "glass": ((150, 196, 220), (104, 150, 182), (178, 214, 232), (214, 234, 244), None),
    "concrete": ((222, 216, 204), (200, 192, 178), (112, 146, 176), (236, 232, 224), (255, 226, 150)),
    "brick": ((204, 132, 100), (176, 104, 76), (86, 102, 122), (226, 196, 176), (255, 220, 140)),
    "white": ((240, 242, 244), (218, 222, 226), (92, 110, 130), (250, 250, 250), None),
    "sand": ((228, 206, 162), (204, 180, 136), (120, 140, 160), (244, 232, 206), (255, 230, 160)),
}


def _pg():
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    if not pygame.display.get_init():
        pygame.display.init()           # no audio: the game is silent
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(len(a)))


class WorldPainter:
    """Draws the world of a Simulator onto a pygame Surface region."""

    def __init__(self, scale: float = 1.0):
        self.pg = _pg()
        self.scale = scale
        self.ppm = PPM * scale
        self.w = int(round(WORLD_PX_W * scale))
        self.h = int(round(WORLD_PX_H * scale))
        self.font_chip = self.pg.font.Font(None, max(10, int(17 * scale)))
        self._sprites: dict = {}
        self._sky = None
        self._clouds = None
        self._hills = None
        self._fog = None
        self._frame_seen = 0

    # ---------------------------------------------------------------- helpers ---------
    def sx(self, sim, wx: float) -> float:
        return (wx - sim.distance_m) * self.ppm

    def sy(self, wy: float) -> float:
        return (C.WORLD_H - wy) * self.ppm

    def m(self, metres: float) -> int:
        return max(1, int(round(metres * self.ppm)))

    def _vgrad(self, w: int, h: int, top, bottom, alpha: bool = True):
        """Vertical gradient surface (fast, via numpy)."""
        pg = self.pg
        t = np.linspace(0.0, 1.0, max(h, 1))[None, :, None]
        col = (np.array(top, float)[None, None, :] * (1 - t) + np.array(bottom, float)[None, None, :] * t)
        arr = np.repeat(col, max(w, 1), axis=0).astype(np.uint8)
        surf = pg.surfarray.make_surface(arr)
        return _to_alpha(pg, surf) if alpha else surf

    def _mask(self, sprite, draw_fn):
        """Keep only the pixels inside the shape drawn by draw_fn(mask_surface)."""
        pg = self.pg
        mask = pg.Surface(sprite.get_size(), pg.SRCALPHA)
        draw_fn(mask)
        sprite.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
        return sprite

    # ---------------------------------------------------------------- background ------
    def _background(self):
        pg = self.pg
        if self._sky is None:
            sky = self._vgrad(self.w, self.h, SKY_TOP, SKY_BOTTOM, alpha=False)
            glow = pg.Surface((self.w, self.h), pg.SRCALPHA)
            cx, cy = int(self.w * 0.84), int(self.h * 0.16)
            for r, a in ((0.20, 18), (0.13, 30), (0.08, 60), (0.045, 150)):
                pg.draw.circle(glow, (*SUN, a), (cx, cy), int(self.h * r))
            pg.draw.circle(glow, (255, 252, 236, 255), (cx, cy), int(self.h * 0.032))
            sky.blit(glow, (0, 0))
            self._sky = sky
            # decorative clouds (white, soft), on a strip three screens wide
            rnd = random.Random(7)
            clouds = []
            period = self.w * 3
            for i in range(8):
                cw = int(self.ppm * rnd.uniform(1.8, 3.6))
                ch = int(cw * rnd.uniform(0.32, 0.45))
                cs = pg.Surface((cw, ch), pg.SRCALPHA)
                for _ in range(7):
                    r = int(ch * rnd.uniform(0.28, 0.48))
                    x = int(rnd.uniform(r, cw - r))
                    y = int(rnd.uniform(ch * 0.45, ch - r))
                    pg.draw.circle(cs, (255, 255, 255, 85), (x, y), r)
                pg.draw.ellipse(cs, (255, 255, 255, 105), (int(cw * 0.1), int(ch * 0.55), int(cw * 0.8), int(ch * 0.4)))
                clouds.append((rnd.uniform(0, period), rnd.uniform(0.12, 0.40) * self.h, cs))
            self._clouds = (period, clouds)
            # far hills: a seamless strip two screens wide (period = 2 w)
            hw = self.w * 2
            hh = int(self.ppm * 1.5)
            # opaque strip: the sky behind the hills is blended in, so the two copies that
            # tile the scroll join without a seam
            y0 = self.h - hh - self.m(0.12)
            hills = pg.Surface((hw, hh))
            for x in range(0, hw, self.w):
                hills.blit(sky, (x, 0), area=pg.Rect(0, y0, self.w, hh))
            for col, amp, base, f1, f2, ph in ((HILL_FAR, 0.55, 0.45, 2, 5, 0.3), (HILL_NEAR, 0.4, 0.25, 3, 7, 1.7)):
                pts = [(0, hh)]
                xs = list(range(0, hw, max(2, hw // 240))) + [hw]     # end exactly on the period
                for x in xs:
                    u = 2 * math.pi * x / hw
                    y = hh * (1 - base - amp * (0.5 + 0.3 * math.sin(f1 * u + ph) + 0.2 * math.sin(f2 * u + 2 * ph)))
                    pts.append((x, y))
                pts.append((hw, hh))
                pg.draw.polygon(hills, (*col, 150), pts)
            self._hills = hills
        return self._sky

    def _draw_background(self, surf, sim, ox, oy):
        pg = self.pg
        surf.blit(self._background(), (ox, oy))
        period, clouds = self._clouds
        off = (sim.distance_m * self.ppm * 0.12) % period
        for cx, cy, cs in clouds:
            x = (cx - off) % period - cs.get_width()
            if -cs.get_width() < x < self.w:
                surf.blit(cs, (ox + x, oy + cy))
        hw = self._hills.get_width()
        off = (sim.distance_m * self.ppm * 0.3) % hw
        y = oy + self.h - self._hills.get_height() - self.m(0.12)
        surf.blit(self._hills, (ox - off, y))
        surf.blit(self._hills, (ox - off + hw, y))

    def draw_empty(self, surf, ox: int = 0, oy: int = 0) -> None:
        """Sky and ground only (title screen)."""
        surf.blit(self._background(), (ox, oy))
        self._draw_ground(surf, ox, oy)

    def _draw_ground(self, surf, ox, oy):
        pg = self.pg
        gh = self.m(0.16)
        key = ("ground", gh)
        g = self._sprites.get(key)
        if g is None:
            g = self._vgrad(self.w, gh, GRASS_TOP, GRASS_BOTTOM, alpha=False)
            pg.draw.rect(g, SOIL, (0, gh - max(2, gh // 4), self.w, max(2, gh // 4)))
            self._sprites[key] = g
        surf.blit(g, (ox, oy + self.h - gh))
        # sky ceiling: a faint dotted line and a soft band
        step = self.m(0.4)
        for x in range(0, self.w, step):
            pg.draw.line(surf, (122, 160, 200), (ox + x, oy + 1), (ox + x + step // 2, oy + 1), max(1, int(self.scale)))

    # ---------------------------------------------------------------- sprites ---------
    def _sprite(self, h):
        key = h.key()
        spr = self._sprites.get(key)
        if spr is None:
            if h.kind == C.BUILDING:
                spr = self._building(h)
            elif h.kind == C.MOUNTAIN:
                spr = self._mountain(h)
            elif h.kind == C.STORM:
                spr = self._storm(h)
            elif h.parts:
                spr = self._parts_sprite(h)
            else:
                spr = self._downdraft(h)
            self._sprites[key] = spr
        return spr

    def _building(self, h):
        pg = self.pg
        W, H = self.m(h.x1 - h.x0), self.m(h.y1)
        rnd = random.Random(h.uid * 7919 + int(h.y1 * 100))
        style = rnd.choice(list(BUILDING_STYLES))
        top, bottom, win, frame, lit = BUILDING_STYLES[style]
        s = self._vgrad(W, H, top, bottom)
        if style == "glass":
            step = self.m(0.32)
            for x in range(step, W, step):
                pg.draw.line(s, frame, (x, 0), (x, H), max(1, int(self.scale)))
            fl = self.m(0.45)
            for y in range(fl, H, fl):
                pg.draw.line(s, (*_lerp(frame, bottom, 0.4),), (0, y), (W, y), max(1, int(self.scale)))
            refl = pg.Surface((W, H), pg.SRCALPHA)
            pg.draw.polygon(refl, (255, 255, 255, 46), [(W * 0.15, 0), (W * 0.45, 0), (W * 0.05, H * 0.6), (0, H * 0.6), (0, H * 0.35)])
            s.blit(refl, (0, 0))
        elif style == "white":
            fl = self.m(0.5)
            band = max(2, int(fl * 0.42))
            for y in range(self.m(0.25), H - band, fl):
                pg.draw.rect(s, win, (self.m(0.1), y, W - 2 * self.m(0.1), band))
                pg.draw.line(s, (150, 170, 190), (self.m(0.1), y), (W - self.m(0.1), y), 1)
        else:
            ww, wh = self.m(0.2), self.m(0.27)
            gx, gy = self.m(0.38), self.m(0.52)
            margin = max(self.m(0.16), (W - ((W - 2 * self.m(0.16)) // gx) * gx + gx - ww) // 2)
            for y in range(self.m(0.3), H - wh - self.m(0.05), gy):
                for x in range(margin, W - ww - self.m(0.08) + 1, gx):
                    c = lit if (lit is not None and rnd.random() < 0.12) else win
                    pg.draw.rect(s, frame, (x - 1, y - 1, ww + 2, wh + 2))
                    pg.draw.rect(s, c, (x, y, ww, wh))
                    pg.draw.line(s, _lerp(c, (255, 255, 255), 0.35), (x, y), (x + ww - 1, y), 1)
        # 3D: shaded right side, parapet, ground-floor entrance
        shade = pg.Surface((max(1, int(W * 0.14)), H), pg.SRCALPHA)
        shade.fill((0, 0, 0, 30))
        s.blit(shade, (W - shade.get_width(), 0))
        pg.draw.rect(s, _lerp(bottom, (60, 60, 60), 0.3), (0, 0, W, max(2, self.m(0.07))))
        if H > self.m(1.2) and W > self.m(1.0):
            dw, dh = self.m(0.38), self.m(0.42)
            pg.draw.rect(s, (70, 80, 92), (W // 2 - dw // 2, H - dh, dw, dh))
        pg.draw.rect(s, (*_lerp(bottom, (40, 40, 40), 0.45), 255), (0, 0, W, H), max(1, int(self.scale)))
        return s

    def _mountain(self, h):
        pg = self.pg
        W, H = self.m(h.x1 - h.x0), self.m(h.y1)
        xc = (h.xc - h.x0) * self.ppm
        rnd = random.Random(h.uid * 104729)
        rock_top, rock_mid, grass = (196, 194, 184), (156, 158, 140), (128, 168, 96)
        s = self.pg.Surface((W, H), pg.SRCALPHA)
        g1 = self._vgrad(W, int(H * 0.62) + 1, rock_top, rock_mid)
        g2 = self._vgrad(W, H - int(H * 0.62) + 1, rock_mid, grass)
        s.blit(g1, (0, 0))
        s.blit(g2, (0, int(H * 0.62)))
        # shaded right face
        face = pg.Surface((W, H), pg.SRCALPHA)
        pg.draw.polygon(face, (40, 46, 58, 52), [(xc, 0), (W, H), (xc + (W - xc) * 0.12, H)])
        s.blit(face, (0, 0))
        # rock strata
        for _ in range(int(6 + W / self.ppm)):
            y = rnd.uniform(0.25, 0.85) * H
            x = rnd.uniform(0.1, 0.9) * W
            L = rnd.uniform(0.3, 0.9) * self.ppm
            pg.draw.line(s, (120, 122, 110, 120), (x, y), (x + L, y + L * rnd.uniform(-0.25, 0.25)), max(1, int(self.scale)))
        # snow cap with a jagged lower edge
        depth = H * rnd.uniform(0.17, 0.24)
        pts = [(xc, 0)]
        n = 7
        for i in range(n + 1):
            f = i / n
            yy = depth * (0.75 + 0.35 * rnd.random())
            pts.append((xc + (f - 0.5) * 2 * (xc if f < 0.5 else W - xc) * yy / H, yy))
        pts.append((xc, 0))
        pg.draw.polygon(s, (250, 252, 253), pts)
        right = [(xc, 0)] + [p for p in pts if p[0] >= xc]
        if len(right) >= 3:
            pg.draw.polygon(s, (214, 226, 238), right)
        # keep only the triangle
        tri = [(0, H), (xc, 0), (W, H)]
        self._mask(s, lambda mk: pg.draw.polygon(mk, (255, 255, 255, 255), tri))
        edge = (110, 118, 104)
        pg.draw.line(s, edge, (0, H), (xc, 0), max(1, int(self.scale)))
        pg.draw.line(s, edge, (xc, 0), (W, H), max(1, int(self.scale)))
        return s

    def _storm(self, h):
        pg = self.pg
        W = self.m(h.x1 - h.x0)
        D = self.m(C.CEILING - h.y0)
        r = h.r * self.ppm
        pad = self.m(0.45)
        s = self._vgrad(W, D + pad, STORM_TOP, STORM_BOTTOM)
        rnd = random.Random(h.uid * 15485863)
        # volume: lighter and darker puffs inside the cloud
        vol = pg.Surface((W, D + pad), pg.SRCALPHA)
        for _ in range(int(10 + W / self.ppm * 3)):
            cx = rnd.uniform(0, W)
            cy = rnd.uniform(0, D)
            rr = rnd.uniform(0.3, 0.7) * self.ppm
            pg.draw.circle(vol, (255, 255, 255, 34) if rnd.random() < 0.6 else (40, 46, 60, 26), (cx, cy), rr)
        s.blit(vol, (0, 0))
        trap = [(0, 0), (r, D), (W - r, D), (W, 0)]
        rb = 0.2 * self.ppm                     # puff radius: the puffs reach the collision edge, never past it

        def shape(mk):
            # body inset by one puff radius, so the drawn cloud never hangs below its hitbox (v2.1: the
            # owner saw the drone touch the cloud's bottom without a crash; the puffs used to be centred
            # on the collision edge and stuck out 0.2-0.28 m)
            if D > rb:
                inner = [(0, 0), (r * (D - rb) / D, D - rb), (W - r * (D - rb) / D, D - rb), (W, 0)]
            else:
                inner = [(0, 0), (W, 0)]
            if len(inner) > 2:
                pg.draw.polygon(mk, (255, 255, 255, 255), inner)
            # puffy edge: overlapping bulges centred one radius inside the collision edge
            for (ax, ay), (bx, by) in ((trap[0], trap[1]), (trap[1], trap[2]), (trap[2], trap[3])):
                L = math.hypot(bx - ax, by - ay)
                if L < 1e-6:
                    continue
                nx, ny = (by - ay) / L, -(bx - ax) / L          # unit normal pointing into the cloud
                if nx * (W / 2 - ax) + ny * (D / 2 - ay) < 0:
                    nx, ny = -nx, -ny
                n = max(2, int(L / (0.22 * self.ppm)))
                for i in range(n + 1):
                    f = i / n
                    rr = rb * (1.0 - 0.15 * ((i * 7) % 3) / 2)
                    pg.draw.circle(mk, (255, 255, 255, 255),
                                   (ax + (bx - ax) * f + nx * rr, ay + (by - ay) * f + ny * rr), rr)
        self._mask(s, shape)
        # rain under the flat bottom
        for _ in range(int(W / self.ppm * 5)):
            x = rnd.uniform(r, W - r)
            y = D + rnd.uniform(0.05, 0.2) * self.ppm
            L = rnd.uniform(0.12, 0.25) * self.ppm
            pg.draw.line(s, (*RAIN, 150), (x, y), (x - L * 0.25, y + L), max(1, int(self.scale)))
        return s

    # ---------------------------------------------------------------- v2.0 obstacles --
    def _parts_sprite(self, h):
        """Full-height sprite of a parts-built obstacle; its left edge is PART_MARGINS[kind][0] m
        left of h.x0 and its top is the sky ceiling."""
        pg = self.pg
        ml, mr = PART_MARGINS[h.kind]
        W = max(1, int(math.ceil((h.x1 - h.x0 + ml + mr) * self.ppm)))
        s = pg.Surface((W, self.h), pg.SRCALPHA)
        X = lambda wx: (wx - h.x0 + ml) * self.ppm          # noqa: E731
        Y = lambda wy: (C.CEILING - wy) * self.ppm          # noqa: E731
        {C.TREE: self._paint_trees, C.POLE: self._paint_poles, C.POWERLINE: self._paint_powerline,
         C.KITE: self._paint_kite, C.CRANE: self._paint_crane}[h.kind](s, h, X, Y)
        return s

    def _lw(self, metres: float, lo: int = 1) -> int:
        return max(lo, int(round(metres * self.ppm)))

    def _paint_trees(self, s, h, X, Y):
        pg = self.pg
        rnd = random.Random(h.uid * 7907)
        trees = [h.parts[i:i + 4] for i in range(0, len(h.parts), 4)]     # trunk, canopy bottom / middle / top
        for t, *_ in trees:
            x0, x1, top, bot = X(t[1]), X(t[3]), Y(t[4]) - self.m(0.3), Y(t[2])
            w = max(2, int(x1 - x0))
            bark = self._vgrad(w, max(1, int(bot - top)), _lerp(BARK, (0, 0, 0), 0.1), BARK)
            s.blit(bark, (x0, top))
            pg.draw.rect(s, _lerp(BARK, (0, 0, 0), 0.35), (x0 + w * 0.62, top, max(1, w * 0.38), bot - top))
            flare = self.m(0.12)
            pg.draw.polygon(s, BARK, [(x0 - flare, bot), (x0, bot - self.m(0.25)), (x1, bot - self.m(0.25)), (x1 + flare, bot)])
        for _t, *boxes in trees:
            dark, mid, light = LEAVES[rnd.randrange(len(LEAVES))]
            bx0, bx1 = min(X(b[1]) for b in boxes), max(X(b[3]) for b in boxes)
            by0, by1 = min(Y(b[4]) for b in boxes), max(Y(b[2]) for b in boxes)
            cw, ch = bx1 - bx0, by1 - by0
            pad = self.m(0.4)
            lay = pg.Surface((int(cw + 2 * pad), int(ch + 2 * pad)), pg.SRCALPHA)
            ox_, oy_ = pad - bx0, pad - by0
            rr0 = self._lw(0.14)
            for b in boxes:
                pg.draw.rect(lay, mid, (X(b[1]) + ox_, Y(b[4]) + oy_, X(b[3]) - X(b[1]), Y(b[2]) - Y(b[4])), border_radius=rr0)
            # leafy outline: bulges centred on the collision edges, so the drawn tree covers its hitbox
            for b in boxes:
                x0, x1, top, bot = X(b[1]) + ox_, X(b[3]) + ox_, Y(b[4]) + oy_, Y(b[2]) + oy_
                for (ax, ay), (cx_, cy_) in (((x0, top), (x1, top)), ((x0, bot), (x1, bot)), ((x0, top), (x0, bot)),
                                             ((x1, top), (x1, bot))):
                    L = math.hypot(cx_ - ax, cy_ - ay)
                    n = max(1, int(L / (0.4 * self.ppm)))
                    for i in range(n + 1):
                        f = i / n
                        py = ay + (cy_ - ay) * f
                        shade = mid if py < oy_ + by0 + ch * 0.55 + (pad - oy_) * 0 else _lerp(mid, dark, 0.45)
                        pg.draw.circle(lay, shade, (ax + (cx_ - ax) * f, py), self.ppm * rnd.uniform(0.2, 0.3))
            # depth: dark lower right, light upper left, leaf clusters
            for _ in range(int(8 + cw * ch / (self.ppm * self.ppm) * 3)):
                fx, fy = rnd.uniform(0.12, 0.88), rnd.uniform(0.1, 0.9)
                px, py = pad + fx * cw, pad + fy * ch
                k = fx + fy
                shade = dark if k > 1.2 else (light if k < 0.75 else mid)
                pg.draw.circle(lay, (*shade, 115), (px, py), self.ppm * rnd.uniform(0.14, 0.28))
            s.blit(lay, (bx0 - pad, by0 - pad))

    def _paint_poles(self, s, h, X, Y):
        pg = self.pg
        for i in range(0, len(h.parts), 3):
            pole, arm, lamp = h.parts[i:i + 3]
            x0, x1, top, bot = X(pole[1]), X(pole[3]), Y(pole[4]), Y(pole[2])
            w = max(2, int(x1 - x0))
            body = self._vgrad(w, max(1, int(bot - top)), _lerp(POLE_METAL, (255, 255, 255), 0.25), POLE_METAL)
            s.blit(body, (x0, top))
            pg.draw.rect(s, _lerp(POLE_METAL, (0, 0, 0), 0.3), (x0 + w * 0.65, top, max(1, w * 0.35), bot - top))
            pg.draw.rect(s, _lerp(POLE_METAL, (0, 0, 0), 0.2), (x0 - self.m(0.06), bot - self.m(0.35), w + 2 * self.m(0.06), self.m(0.35)))
            pg.draw.rect(s, _lerp(POLE_METAL, (0, 0, 0), 0.1), (X(arm[1]), Y(arm[4]), X(arm[3]) - X(arm[1]), Y(arm[2]) - Y(arm[4])),
                         border_radius=self._lw(0.05))
            lx0, lx1, lt, lb = X(lamp[1]), X(lamp[3]), Y(lamp[4]), Y(lamp[2])
            pg.draw.rect(s, (70, 74, 84), (lx0, lt, lx1 - lx0, lb - lt), border_top_left_radius=self._lw(0.08),
                         border_top_right_radius=self._lw(0.08))
            pg.draw.rect(s, LAMP_GLOW, (lx0 + self._lw(0.03), lb - self._lw(0.06, 2), lx1 - lx0 - 2 * self._lw(0.03), self._lw(0.06, 2)))
            glow = pg.Surface((int(lx1 - lx0 + self.m(0.6)), self.m(0.5)), pg.SRCALPHA)
            pg.draw.polygon(glow, (*LAMP_GLOW, 60), [(self.m(0.3), 0), (glow.get_width() - self.m(0.3), 0),
                                                     (glow.get_width(), glow.get_height()), (0, glow.get_height())])
            s.blit(glow, (lx0 - self.m(0.3), lb))

    def _back_pole(self, s, x, top, X, Y, crossarm: float):
        """A wooden pole beside the road (background: lighter, not an obstacle)."""
        pg = self.pg
        w = self._lw(0.16, 2)
        col = (*BACK_POLE, 200)
        pg.draw.rect(s, col, (X(x) - w / 2, Y(top + 0.35), w, Y(0) - Y(top + 0.35)))
        pg.draw.rect(s, col, (X(x - crossarm / 2), Y(top + 0.2), X(x + crossarm / 2) - X(x - crossarm / 2), self._lw(0.08, 2)))

    def _paint_powerline(self, s, h, X, Y):
        pg = self.pg
        top = h.y1
        for x in (h.x0, h.x1):
            self._back_pole(s, x, top, X, Y, 0.9)
        for q in h.parts:
            pg.draw.line(s, WIRE_DARK, (X(q[1]), Y(q[2])), (X(q[3]), Y(q[4])), self._lw(0.07, 2))
        for x in (h.x0, h.x1):          # insulators where the wires meet the poles
            for yy in (top, top - C.HZ["powerline"]["wire_gap"]):
                pg.draw.circle(s, (210, 214, 220), (X(x), Y(yy)), self._lw(0.07, 2))

    def _paint_kite(self, s, h, X, Y):
        pg = self.pg
        rnd = random.Random(h.uid * 3571)
        c1, c2 = KITE_COLORS[rnd.randrange(len(KITE_COLORS))]
        edges = h.parts[:4]
        tail, string = h.parts[4], h.parts[5]
        top, right, bot, left = [(X(q[1]), Y(q[2])) for q in edges]
        cx, cy = top[0], left[1]
        # string first (behind), down to a small ground stake
        pg.draw.line(s, (90, 90, 96), (X(string[1]), Y(string[2])), (X(string[3]), Y(string[4])), self._lw(0.03, 1))
        ax = X(string[3])
        pg.draw.rect(s, (110, 86, 60), (ax - self._lw(0.04, 2), Y(0.25), self._lw(0.08, 3), Y(0) - Y(0.25)))
        for i, (a, b) in enumerate(((top, right), (right, bot), (bot, left), (left, top))):
            pg.draw.polygon(s, c1 if i % 2 == 0 else c2, [a, b, (cx, cy)])
        pg.draw.polygon(s, _lerp(c1, (0, 0, 0), 0.45), [top, right, bot, left], self._lw(0.03, 1))
        pg.draw.line(s, (60, 50, 40), top, bot, self._lw(0.025, 1))
        pg.draw.line(s, (60, 50, 40), left, right, self._lw(0.025, 1))
        tx0, ty0, tx1, ty1 = X(tail[1]), Y(tail[2]), X(tail[3]), Y(tail[4])
        pg.draw.line(s, (90, 90, 96), (tx0, ty0), (tx1, ty1), self._lw(0.02, 1))
        for f in (0.35, 0.7, 1.0):      # bows along the tail
            bx, by = tx0 + (tx1 - tx0) * f, ty0 + (ty1 - ty0) * f
            b = self.m(0.07)
            col = c2 if f != 0.7 else c1
            pg.draw.polygon(s, col, [(bx - b, by - b * 0.6), (bx, by), (bx - b, by + b * 0.6)])
            pg.draw.polygon(s, col, [(bx + b, by - b * 0.6), (bx, by), (bx + b, by + b * 0.6)])

    def _paint_crane(self, s, h, X, Y):
        pg = self.pg
        jib, cable, hook = h.parts
        jx0, jx1, jt, jb = X(jib[1]), X(jib[3]), Y(jib[4]), Y(jib[2])
        # tower, cab, counter-jib and counterweight beside the road (background)
        tx = h.x1 + 0.35
        tw = 0.8
        back = (*_lerp(CRANE_YELLOW, (255, 255, 255), 0.45), 190)
        bline = (*_lerp(CRANE_DARK, (255, 255, 255), 0.4), 190)
        lay = pg.Surface(s.get_size(), pg.SRCALPHA)
        pg.draw.rect(lay, back, (X(tx - tw / 2), Y(jib[4] + 1.3), X(tx + tw / 2) - X(tx - tw / 2), Y(0) - Y(jib[4] + 1.3)))
        step = 0.8
        yy = 0.0
        while yy < jib[4] + 1.2:
            pg.draw.line(lay, bline, (X(tx - tw / 2), Y(yy)), (X(tx + tw / 2), Y(yy + step)), self._lw(0.03, 1))
            pg.draw.line(lay, bline, (X(tx - tw / 2), Y(yy + step)), (X(tx + tw / 2), Y(yy + step)), self._lw(0.03, 1))
            yy += step
        pg.draw.rect(lay, back, (X(tx), jt, X(h.x1 + 3.9) - X(tx), jb - jt))                  # counter-jib
        pg.draw.rect(lay, (170, 170, 168, 200), (X(h.x1 + 2.9), jb, X(h.x1 + 3.9) - X(h.x1 + 2.9), self.m(0.9)))
        pg.draw.rect(lay, (*_lerp(CRANE_DARK, (255, 255, 255), 0.3), 200), (X(tx - 0.95), jb, self.m(0.6), self.m(0.6)))   # cab
        pg.draw.line(lay, bline, (X(tx), Y(jib[4] + 1.3)), (X(jib[1] + 0.3 * (h.x1 - jib[1])), jt), self._lw(0.025, 1))
        pg.draw.line(lay, bline, (X(tx), Y(jib[4] + 1.3)), (X(h.x1 + 3.8), jt), self._lw(0.025, 1))
        s.blit(lay, (0, 0))
        # jib: a yellow lattice truss over the road
        pg.draw.rect(s, CRANE_YELLOW, (jx0, jt, jx1 - jx0, jb - jt))
        th = self._lw(0.07, 2)
        pg.draw.rect(s, CRANE_DARK, (jx0, jt, jx1 - jx0, th))
        pg.draw.rect(s, CRANE_DARK, (jx0, jb - th, jx1 - jx0, th))
        d = jb - jt
        x = jx0
        flip = False
        while x < jx1 - 1:
            xe = min(x + d, jx1)
            pg.draw.line(s, CRANE_DARK, (x, jb if flip else jt), (xe, jt if flip else jb), self._lw(0.04, 1))
            x, flip = xe, not flip
        # trolley, cable and hook block
        hx = X(cable[1])
        pg.draw.rect(s, (70, 74, 84), (hx - self.m(0.22), jb, self.m(0.44), self.m(0.14)))
        pg.draw.line(s, WIRE_DARK, (hx, jb), (hx, Y(cable[2])), self._lw(0.04, 1))
        hx0, hx1, ht, hb = X(hook[1]), X(hook[3]), Y(hook[4]), Y(hook[2])
        pg.draw.rect(s, CRANE_YELLOW, (hx0, ht, hx1 - hx0, (hb - ht) * 0.6), border_radius=self._lw(0.05))
        for k in range(3):
            pg.draw.line(s, (40, 40, 40), (hx0 + (hx1 - hx0) * k / 3, ht), (hx0 + (hx1 - hx0) * (k + 1) / 3, ht + (hb - ht) * 0.6),
                         self._lw(0.04, 1))
        cxh, r = (hx0 + hx1) / 2, (hb - ht) * 0.22
        pg.draw.arc(s, (60, 62, 68), (cxh - r, hb - 2 * r, 2 * r, 2 * r), math.pi, 2 * math.pi + 1.2, self._lw(0.05, 2))

    def _downdraft(self, h):
        pg = self.pg
        W = self.m(h.x1 - h.x0)
        s = pg.Surface((W, self.h), pg.SRCALPHA)
        s.fill((*DOWNDRAFT, 40))
        for x in (0, W - 1):
            pg.draw.line(s, (*DOWNDRAFT, 90), (x, 0), (x, self.h), max(1, int(self.scale)))
        return s

    # ---------------------------------------------------------------- frame -----------
    def draw(self, surf, sim, ox: int = 0, oy: int = 0, attended_uid: int | None = None,
             show_wind: bool = False, show_chips: bool = True) -> None:
        """show_wind: the gust arrow beside the drone (off since v2.0, owner request; the GUST chip
        still shows the wind)."""
        pg = self.pg
        p = self.ppm
        world = pg.Rect(ox, oy, self.w, self.h)
        clip = surf.get_clip()
        surf.set_clip(world)
        self._draw_background(surf, sim, ox, oy)
        t = sim.frame * C.DT
        att_rect = None
        visible = []
        for h in sim.statics:
            x0 = ox + self.sx(sim, h.x0)
            x1 = ox + self.sx(sim, h.x1)
            ml, mr = PART_MARGINS.get(h.kind, (0.0, 0.0))
            if x1 + mr * p < ox - 4 or x0 - ml * p > ox + self.w + 4:
                continue
            visible.append((h, x0, x1))
        # downdrafts behind everything, with falling air streaks
        for h, x0, x1 in visible:
            if h.kind != C.DOWNDRAFT:
                continue
            surf.blit(self._sprite(h), (x0, oy))
            step = self.m(0.9)
            off = int((t * 4.0 * p) % step)
            for xx in range(int(x0) + self.m(0.4), int(x1) - self.m(0.2), self.m(0.8)):
                for yy in range(-step + off, self.h, step):
                    pg.draw.line(surf, (255, 255, 255), (xx, oy + yy), (xx - self.m(0.05), oy + yy + self.m(0.35)),
                                 max(1, int(1.5 * self.scale)))
            if attended_uid is not None and abs(h.uid) == attended_uid:
                att_rect = pg.Rect(x0, oy, x1 - x0, self.h)
        for h, x0, x1 in visible:
            k = h.kind
            if k == C.DOWNDRAFT:
                continue
            spr = self._sprite(h)
            if h.parts:
                surf.blit(spr, (x0 - PART_MARGINS[k][0] * p, oy))
                rect = pg.Rect(x0, oy + self.sy(h.y1), x1 - x0, (h.y1 - h.y0) * p)
            elif k == C.STORM:
                surf.blit(spr, (x0, oy))
                rect = pg.Rect(x0, oy, x1 - x0, (C.CEILING - h.y0) * p)
            else:
                top = oy + self.sy(h.y1)
                surf.blit(spr, (x0, top))
                rect = pg.Rect(x0, top, x1 - x0, h.y1 * p)
            if attended_uid is not None and abs(h.uid) == attended_uid:
                att_rect = rect
        for mv in sim.movers:
            x0w = mv.xr - mv.s * (t - mv.tr)
            x0 = ox + self.sx(sim, x0w)
            if x0 > ox + self.w + 4 or x0 + mv.w * p < ox - 4:
                continue
            my = mv.y_at(t)
            r = pg.Rect(x0, oy + self.sy(my + mv.h / 2), mv.w * p, mv.h * p)
            if mv.kind == C.BIRD:
                self._bird(surf, r, mv, t)
            elif mv.kind == C.PLANE:
                self._plane(surf, r, mv, t)
            elif mv.sub == 1:
                self._balloon(surf, r, mv)
            else:
                self._traffic_drone(surf, r, mv, t)
            if attended_uid is not None and abs(mv.uid) == attended_uid:
                att_rect = r
        self._draw_ground(surf, ox, oy)
        self._drone_shadow(surf, sim, ox, oy)
        if att_rect is not None:
            pg.draw.rect(surf, ATTENDED, att_rect.inflate(int(0.25 * p), int(0.25 * p)), max(2, int(2 * self.scale)),
                         border_radius=max(2, int(0.12 * p)))
        # fog: light haze everywhere, dense beyond the 8 m sensor range
        if sim.fog_end > sim.frame:
            if self._fog is None or self._fog.get_size() != (self.w, self.h):
                self._fog = pg.Surface((self.w, self.h), pg.SRCALPHA)
            self._fog.fill((*FOG, 64))
            edge = int(self.sx(sim, sim.distance_m + C.DRONE_X + C.HALF_W + C.FOG_RANGE))
            ramp = self.m(1.2)
            for i in range(ramp):
                pg.draw.line(self._fog, (*FOG, 64 + int(120 * i / ramp)), (edge + i, 0), (edge + i, self.h))
            self._fog.fill((*FOG, 184), (edge + ramp, 0, self.w, self.h))
            surf.blit(self._fog, (ox, oy))
        self._draw_drone(surf, sim, ox, oy)
        if show_wind and sim.gust_a != 0.0:
            self._draw_wind(surf, sim, ox, oy)
        if show_chips:
            self._chips(surf, sim, ox, oy)
        surf.set_clip(clip)
        # forget sprites of hazards that left the screen
        self._frame_seen += 1
        if self._frame_seen % 60 == 0 and len(self._sprites) > 64:
            live = {h.key() for h in sim.statics}
            self._sprites = {k: v for k, v in self._sprites.items() if k in live or (isinstance(k, tuple) and k and k[0] == "ground")}

    # ---------------------------------------------------------------- movers ----------
    def _bird(self, surf, r, mv, t):
        pg = self.pg
        p = self.ppm
        col = BIRD_COLORS[abs(mv.uid) % len(BIRD_COLORS)]
        cx, cy = r.centerx, r.centery
        flap = math.sin(t * 11.0 + mv.phase)
        span = 0.36 * p
        lw = max(2, int(2.2 * self.scale))
        # far wing (lighter), body, near wing
        far = _lerp(col, (200, 205, 210), 0.35)
        tip_far = (cx + span * 0.35, cy - span * 0.55 * flap - span * 0.15)
        pg.draw.polygon(surf, far, [(cx - 0.04 * p, cy), (cx + 0.08 * p, cy), tip_far])
        pg.draw.ellipse(surf, col, (cx - 0.17 * p, cy - 0.06 * p, 0.34 * p, 0.12 * p))
        pg.draw.circle(surf, col, (int(cx - 0.17 * p), int(cy - 0.035 * p)), max(2, int(0.06 * p)))
        pg.draw.polygon(surf, (240, 160, 40), [(cx - 0.22 * p, cy - 0.04 * p), (cx - 0.3 * p, cy - 0.02 * p),
                                               (cx - 0.22 * p, cy - 0.005 * p)])
        pg.draw.polygon(surf, col, [(cx + 0.15 * p, cy - 0.02 * p), (cx + 0.27 * p, cy - 0.06 * p), (cx + 0.26 * p, cy + 0.03 * p)])
        tip = (cx - span * 0.15, cy - span * flap - 0.02 * p)
        pg.draw.polygon(surf, col, [(cx - 0.07 * p, cy - 0.01 * p), (cx + 0.1 * p, cy - 0.01 * p), tip])
        pg.draw.line(surf, _lerp(col, (0, 0, 0), 0.3), (cx - 0.07 * p, cy - 0.01 * p), tip, max(1, lw // 2))

    def _traffic_drone(self, surf, r, mv, t):
        pg = self.pg
        p = self.ppm
        body = pg.Rect(r.left + 0.12 * p, r.top + 0.16 * p, r.w - 0.24 * p, r.h - 0.2 * p)
        arm_y = r.top + 0.12 * p
        lw = max(1, int(2 * self.scale))
        for ax in (r.left + 0.16 * p, r.right - 0.16 * p):
            pg.draw.line(surf, TRAFFIC_DARK, (ax, arm_y), (body.centerx, body.top + 2), lw)
            pg.draw.rect(surf, TRAFFIC_DARK, (ax - 0.03 * p, arm_y - 0.02 * p, 0.06 * p, 0.08 * p))
            disc = pg.Surface((int(0.34 * p), max(2, int(0.07 * p))), pg.SRCALPHA)
            pg.draw.ellipse(disc, (*ROTOR_BLUR, 110), disc.get_rect())
            surf.blit(disc, (ax - disc.get_width() / 2, arm_y - disc.get_height() / 2 - 0.02 * p))
        pg.draw.rect(surf, TRAFFIC_BODY, body, border_radius=max(2, int(0.08 * p)))
        pg.draw.rect(surf, _lerp(TRAFFIC_BODY, (0, 0, 0), 0.25), body, max(1, int(self.scale)),
                     border_radius=max(2, int(0.08 * p)))
        pg.draw.line(surf, (255, 255, 255), (body.left + 3, body.top + 2), (body.right - 3, body.top + 2), 1)
        pg.draw.circle(surf, TRAFFIC_DARK, (int(body.left + 0.12 * p), int(body.bottom)), max(2, int(0.06 * p)))
        on = int(t * 2) % 2 == 0
        pg.draw.circle(surf, (230, 60, 50) if on else (150, 70, 70), (int(body.left + 2), int(body.centery)), max(2, int(0.035 * p)))
        pg.draw.circle(surf, (60, 200, 90) if not on else (70, 130, 80), (int(body.right - 2), int(body.centery)), max(2, int(0.035 * p)))

    def _plane(self, surf, r, mv, t):
        """Side view of a plane flying left (toward our drone). The whole drawing stays inside
        the hitbox rectangle r: fuselage, tail fin, near wing and engine or propeller."""
        pg = self.pg
        key = ("plane", mv.sub, abs(mv.uid) % len(PLANE_LIVERIES), r.w, r.h)
        spr = self._sprites.get(key)
        if spr is None:
            spr = self._plane_sprite(mv.sub, PLANE_LIVERIES[abs(mv.uid) % len(PLANE_LIVERIES)], r.w, r.h)
            self._sprites[key] = spr
        surf.blit(spr, r.topleft)
        p = self.ppm
        if mv.sub == 1:     # spinning propeller at the nose
            ph = (t * 40.0) % 2.0
            blade = r.h * (0.18 + 0.22 * abs(ph - 1.0))
            nx, ny = r.left + 0.03 * r.w, r.top + r.h * 0.56
            disc = pg.Surface((max(2, int(0.08 * p)), max(4, int(r.h * 0.9))), pg.SRCALPHA)
            pg.draw.ellipse(disc, (*ROTOR_BLUR, 90), disc.get_rect())
            surf.blit(disc, (nx - disc.get_width() / 2, ny - disc.get_height() / 2))
            pg.draw.line(surf, (60, 64, 72), (nx, ny - blade), (nx, ny + blade), max(2, int(2 * self.scale)))
        on = int(t * 1.6) % 2 == 0          # red beacon on the belly, white strobe on the tail
        rad = max(2, int(0.04 * p))
        pg.draw.circle(surf, (235, 60, 50) if on else (150, 80, 80), (int(r.left + r.w * 0.45), int(r.bottom - r.h * 0.12)), rad)
        if not on:
            pg.draw.circle(surf, (255, 255, 255), (int(r.right - r.w * 0.04), int(r.top + r.h * 0.06)), rad)

    def _plane_sprite(self, sub, livery, W, H):
        pg = self.pg
        body, belly, stripe, tail = livery
        spr = pg.Surface((max(1, W), max(1, H)), pg.SRCALPHA)
        line = _lerp(belly, (0, 0, 0), 0.35)
        lw = max(1, int(self.scale))
        if sub == 0:
            # airliner: fuselage in the middle, tall swept fin at the back, engine under the wing
            ft, fb = H * 0.38, H * 0.78
            fh = fb - ft
            fin = [(W * 0.76, ft + 1), (W * 0.88, 0), (W * 0.97, 0), (W * 0.985, ft + fh * 0.3)]
            pg.draw.polygon(spr, tail, fin)
            pg.draw.polygon(spr, _lerp(tail, (0, 0, 0), 0.25), fin, lw)
            # far wing (in shade, above the fuselage line)
            pg.draw.polygon(spr, _lerp(belly, (0, 0, 0), 0.22), [(W * 0.40, ft + fh * 0.35), (W * 0.52, ft + fh * 0.35),
                                                                   (W * 0.61, ft - fh * 0.2), (W * 0.58, ft - fh * 0.2)])
            # fuselage: rounded nose, straight cabin, tapered tail cone
            body_pts = [(W * 0.10, ft), (W * 0.86, ft), (W * 0.995, ft + fh * 0.22), (W * 0.995, ft + fh * 0.42),
                        (W * 0.86, fb), (W * 0.10, fb)]
            pg.draw.polygon(spr, body, body_pts)
            pg.draw.ellipse(spr, body, (0, ft, W * 0.22, fh))
            pg.draw.polygon(spr, belly, [(W * 0.11, ft + fh * 0.64), (W * 0.86, ft + fh * 0.64), (W * 0.86, fb), (W * 0.11, fb)])
            pg.draw.polygon(spr, belly, [(W * 0.86, ft + fh * 0.64), (W * 0.995, ft + fh * 0.42), (W * 0.86, fb)])
            pg.draw.arc(spr, belly, (0, ft, W * 0.22, fh), math.pi * 1.05, math.pi * 1.6, max(2, int(fh * 0.3)))
            # outline: top, belly and nose (no seam where the nose meets the cabin)
            pg.draw.line(spr, line, (W * 0.10, ft), (W * 0.86, ft), lw)
            pg.draw.line(spr, line, (W * 0.86, ft), (W * 0.995, ft + fh * 0.22), lw)
            pg.draw.line(spr, line, (W * 0.10, fb - 1), (W * 0.86, fb - 1), lw)
            pg.draw.line(spr, line, (W * 0.86, fb - 1), (W * 0.995, ft + fh * 0.42), lw)
            pg.draw.arc(spr, line, (0, ft, W * 0.22, fh), math.pi * 0.5, math.pi * 1.5, lw)
            # cheat line in the livery colour
            pg.draw.line(spr, stripe, (W * 0.06, ft + fh * 0.6), (W * 0.95, ft + fh * 0.6), max(2, int(fh * 0.13)))
            # cockpit and cabin windows
            pg.draw.polygon(spr, PLANE_WINDOW, [(W * 0.03, ft + fh * 0.36), (W * 0.065, ft + fh * 0.2),
                                                (W * 0.09, ft + fh * 0.2), (W * 0.08, ft + fh * 0.38)])
            n = max(6, int(W / (0.26 * self.ppm)))
            for i in range(n):
                x = W * 0.14 + (W * 0.66) * i / (n - 1)
                pg.draw.circle(spr, PLANE_WINDOW, (int(x), int(ft + fh * 0.32)), max(1, int(fh * 0.075)))
            pg.draw.line(spr, line, (W * 0.13, ft + fh * 0.1), (W * 0.13, ft + fh * 0.62), 1)   # door
            # near wing, swept back below the fuselage, with the engine hanging under it
            wing_c = _lerp(body, (90, 100, 115), 0.35)
            pg.draw.polygon(spr, wing_c, [(W * 0.36, ft + fh * 0.62), (W * 0.52, ft + fh * 0.62),
                                          (W * 0.64, H * 0.99), (W * 0.585, H * 0.99)])
            eng = pg.Rect(W * 0.31, fb - fh * 0.12, W * 0.15, H - fb + fh * 0.12 - 1)
            pg.draw.ellipse(spr, (206, 211, 218), eng)
            pg.draw.ellipse(spr, _lerp(stripe, (255, 255, 255), 0.2), (eng.left + eng.w * 0.25, eng.top, eng.w * 0.5, eng.h * 0.35))
            pg.draw.ellipse(spr, (70, 76, 86), (eng.left, eng.top + eng.h * 0.12, eng.w * 0.16, eng.h * 0.76))
            pg.draw.ellipse(spr, line, eng, lw)
            # tailplane
            pg.draw.polygon(spr, tail, [(W * 0.86, ft + fh * 0.25), (W * 0.995, ft + fh * 0.08), (W * 0.995, ft + fh * 0.26)])
        else:
            # small high-wing propeller plane: wing on top, tail fin, fixed landing gear
            ft, fb = H * 0.34, H * 0.78
            fh = fb - ft
            fus = [(W * 0.05, ft + fh * 0.15), (W * 0.30, ft), (W * 0.70, ft + fh * 0.1), (W * 0.98, ft + fh * 0.2),
                   (W * 0.98, ft + fh * 0.45), (W * 0.70, fb - fh * 0.1), (W * 0.30, fb), (W * 0.05, fb - fh * 0.1)]
            fin = [(W * 0.80, ft + fh * 0.15), (W * 0.92, 0), (W * 0.98, 0), (W * 0.98, ft + fh * 0.3)]
            pg.draw.polygon(spr, tail, fin)
            # wheels and struts
            for wx in (W * 0.26, W * 0.48):
                pg.draw.line(spr, (70, 74, 82), (wx, fb - fh * 0.1), (wx - W * 0.02, H * 0.9), max(1, lw))
                pg.draw.circle(spr, (40, 42, 48), (int(wx - W * 0.02), int(H * 0.9)), max(2, int(H * 0.09)))
            pg.draw.polygon(spr, body, fus)
            pg.draw.polygon(spr, belly, [(W * 0.05, ft + fh * 0.6), (W * 0.98, ft + fh * 0.4), (W * 0.98, ft + fh * 0.45),
                                         (W * 0.70, fb - fh * 0.1), (W * 0.30, fb), (W * 0.05, fb - fh * 0.1)])
            pg.draw.line(spr, stripe, (W * 0.06, ft + fh * 0.52), (W * 0.97, ft + fh * 0.33), max(2, int(fh * 0.14)))
            # cabin windows and the high wing with its strut
            pg.draw.polygon(spr, PLANE_WINDOW, [(W * 0.18, ft + fh * 0.12), (W * 0.30, ft + fh * 0.04),
                                                (W * 0.42, ft + fh * 0.06), (W * 0.42, ft + fh * 0.36), (W * 0.20, ft + fh * 0.36)])
            pg.draw.line(spr, body, (W * 0.30, ft + fh * 0.05), (W * 0.30, ft + fh * 0.36), max(1, lw))
            wing = pg.Rect(W * 0.22, ft - H * 0.08, W * 0.30, H * 0.09)
            pg.draw.rect(spr, _lerp(body, (0, 0, 0), 0.08), wing, border_radius=max(1, int(H * 0.04)))
            pg.draw.rect(spr, stripe, (wing.left, wing.top, wing.w * 0.12, wing.h))
            pg.draw.line(spr, (110, 116, 126), (W * 0.40, ft + fh * 0.6), (W * 0.48, wing.bottom), max(1, lw))
            # tailplane and spinner
            pg.draw.polygon(spr, tail, [(W * 0.82, ft + fh * 0.22), (W * 0.99, ft + fh * 0.12), (W * 0.99, ft + fh * 0.3)])
            pg.draw.ellipse(spr, (60, 64, 72), (0, ft + fh * 0.25, W * 0.07, fh * 0.45))
            pg.draw.polygon(spr, line, fus, lw)
        return spr

    def _balloon(self, surf, r, mv):
        pg = self.pg
        p = self.ppm
        c1, c2 = BALLOONS[abs(mv.uid) % len(BALLOONS)]
        env = pg.Rect(r.left, r.top, r.w, r.h * 0.72)
        bw, bh = r.w * 0.32, r.h * 0.13
        basket = pg.Rect(r.centerx - bw / 2, r.bottom - bh, bw, bh)
        # skirt and ropes
        pg.draw.polygon(surf, _lerp(c1, (0, 0, 0), 0.2), [(env.left + env.w * 0.22, env.bottom - env.h * 0.12),
                                                         (env.right - env.w * 0.22, env.bottom - env.h * 0.12),
                                                         (basket.right - bw * 0.1, basket.top - bh * 0.6),
                                                         (basket.left + bw * 0.1, basket.top - bh * 0.6)])
        for x0, x1 in ((basket.left + 1, env.left + env.w * 0.3), (basket.right - 1, env.right - env.w * 0.3)):
            pg.draw.line(surf, (110, 90, 70), (x0, basket.top), (x1, env.bottom - env.h * 0.1), 1)
        pg.draw.ellipse(surf, c1, env)
        for f in (0.62, 0.3):
            g = pg.Rect(0, 0, env.w * f, env.h)
            g.center = env.center
            pg.draw.ellipse(surf, c2 if f == 0.62 else c1, g)
        hl = pg.Surface((int(env.w * 0.3), int(env.h * 0.42)), pg.SRCALPHA)
        pg.draw.ellipse(hl, (255, 255, 255, 70), hl.get_rect())
        surf.blit(hl, (env.left + env.w * 0.14, env.top + env.h * 0.12))
        pg.draw.ellipse(surf, _lerp(c1, (0, 0, 0), 0.3), env, max(1, int(self.scale)))
        pg.draw.rect(surf, BASKET, basket, border_radius=max(1, int(0.03 * p)))
        pg.draw.line(surf, _lerp(BASKET, (0, 0, 0), 0.3), (basket.left, basket.top + bh * 0.4), (basket.right, basket.top + bh * 0.4), 1)

    # ---------------------------------------------------------------- our drone -------
    def _drone_shadow(self, surf, sim, ox, oy):
        pg = self.pg
        X = sim.distance_m + C.DRONE_X
        k = max(0.0, 1.0 - sim.y / 11.0)
        w = int(C.HIT_W * self.ppm * (0.5 + 0.6 * k))
        hgt = max(2, int(0.1 * self.ppm))
        sh = pg.Surface((w, hgt), pg.SRCALPHA)
        pg.draw.ellipse(sh, (40, 60, 30, int(30 + 70 * k)), sh.get_rect())
        surf.blit(sh, (ox + self.sx(sim, X) - w / 2, oy + self.h - self.m(0.16) - hgt / 2))

    def _draw_drone(self, surf, sim, ox, oy):
        pg = self.pg
        p = self.ppm
        hh = C.HALF_H_TUCK if sim.tuck > 0 else C.HALF_H
        X = sim.distance_m + C.DRONE_X
        r = pg.Rect(ox + self.sx(sim, X - C.HALF_W), oy + self.sy(sim.y + hh), C.HIT_W * p, 2 * hh * p)
        lw = max(2, int(2.2 * self.scale))
        # arms, motors and spinning rotor discs just above the body (v2.1: lowered so the rotors stick out
        # at most ~6 cm above the hitbox, which is the body; they used to reach 13 cm above it)
        arm_y = r.top - 0.02 * p
        for mx in (r.left + 0.1 * p, r.right - 0.1 * p):
            pg.draw.line(surf, DRONE_DARK, (mx, r.top + 2), (mx, arm_y), lw)
            pg.draw.rect(surf, DRONE_DARK, (mx - 0.035 * p, arm_y - 0.03 * p, 0.07 * p, 0.05 * p), border_radius=2)
            disc = pg.Surface((int(0.42 * p), max(3, int(0.08 * p))), pg.SRCALPHA)
            pg.draw.ellipse(disc, (*ROTOR_BLUR, 120), disc.get_rect())
            spin = (sim.frame * 3) % 4
            pg.draw.line(disc, (*DRONE_DARK, 200), (disc.get_width() * (0.2 + 0.15 * spin), disc.get_height() / 2),
                         (disc.get_width() * (0.8 - 0.15 * spin), disc.get_height() / 2), max(1, lw // 2))
            surf.blit(disc, (mx - disc.get_width() / 2, arm_y - disc.get_height() / 2))
        # body with a light top and a darker belly
        rad = max(2, int(0.11 * p))
        pg.draw.rect(surf, DRONE_BODY_BOTTOM, r, border_radius=rad)
        top = pg.Rect(r.left, r.top, r.w, max(2, r.h * 0.55))
        pg.draw.rect(surf, DRONE_BODY_TOP, top, border_top_left_radius=rad, border_top_right_radius=rad)
        pg.draw.rect(surf, (176, 84, 10), r, max(1, int(self.scale)), border_radius=rad)
        # canopy (front faces right) and camera
        cw, ch = r.w * 0.28, max(2, r.h * 0.42)
        pg.draw.rect(surf, (52, 70, 92), (r.right - cw - 0.06 * p, r.top + r.h * 0.18, cw, ch), border_radius=max(1, rad // 2))
        pg.draw.line(surf, (160, 200, 230), (r.right - cw - 0.03 * p, r.top + r.h * 0.24), (r.right - 0.1 * p, r.top + r.h * 0.24), 1)
        pg.draw.circle(surf, DRONE_DARK, (int(r.centerx + 0.08 * p), int(r.bottom - 0.02 * p)), max(2, int(0.05 * p)))
        led = (255, 70, 60) if (sim.frame // 10) % 2 == 0 else (150, 60, 50)
        pg.draw.circle(surf, led, (int(r.left + 0.05 * p), int(r.centery)), max(2, int(0.03 * p)))

    def _draw_wind(self, surf, sim, ox, oy):
        pg = self.pg
        p = self.ppm
        X = sim.distance_m + C.DRONE_X
        cx = ox + self.sx(sim, X - 1.3)
        cy = oy + self.sy(sim.y)
        L = (0.6 + 1.2 * min(abs(sim.gust_a) / 8.0, 1.0)) * p
        d = -1 if sim.gust_a > 0 else 1
        w = max(3, int(3.5 * self.scale))
        for dx, a in ((-0.25 * p, 90), (0.25 * p, 90)):
            pg.draw.line(surf, _lerp(WIND, (255, 255, 255), 0.5), (cx + dx, cy - d * L * 0.35), (cx + dx, cy + d * L * 0.35), max(1, w // 2))
        y0, y1 = cy - d * L / 2, cy + d * L / 2
        pg.draw.line(surf, WIND, (cx, y0), (cx, y1), w)
        hs = 0.2 * p
        pg.draw.polygon(surf, WIND, [(cx, y1 + d * hs * 0.4), (cx - hs, y1 - d * hs * 0.6), (cx + hs, y1 - d * hs * 0.6)])

    def _chips(self, surf, sim, ox, oy):
        pg = self.pg
        chips = []
        if sim.gust_a != 0.0:
            chips.append("GUST " + ("UP" if sim.gust_a > 0 else "DOWN"))
        if sim.surge_end > sim.frame:
            chips.append("SURGE")
        if sim.fog_end > sim.frame:
            chips.append("FOG")
        if sim.dd_a != 0.0:
            chips.append("DOWNDRAFT")
        xx = ox + int(10 * self.scale)
        for c in chips:
            img = self.font_chip.render(c, True, CHIP_FG)
            pad = int(7 * self.scale)
            rr = pg.Rect(xx, oy + int(10 * self.scale), img.get_width() + 2 * pad, img.get_height() + pad)
            bg = pg.Surface(rr.size, pg.SRCALPHA)
            pg.draw.rect(bg, (*CHIP_BG, 225), bg.get_rect(), border_radius=int(8 * self.scale))
            pg.draw.rect(bg, (*CHIP_EDGE, 255), bg.get_rect(), max(1, int(self.scale)), border_radius=int(8 * self.scale))
            surf.blit(bg, rr.topleft)
            surf.blit(img, (rr.x + pad, rr.y + pad // 2 + 1))
            xx = rr.right + int(6 * self.scale)


def _to_alpha(pg, surf):
    out = pg.Surface(surf.get_size(), pg.SRCALPHA)
    out.blit(surf, (0, 0))
    return out


class Renderer:
    """Used by DroneDashEnv.render(): 'rgb_array' (offscreen) or 'human' (simple window)."""

    def __init__(self, window: bool = False, scale: float = 1.0):
        if not window and "SDL_VIDEODRIVER" not in os.environ and not os.environ.get("DISPLAY") \
                and os.name != "nt" and not _is_macos():
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        self.pg = _pg()
        self.painter = WorldPainter(scale)
        self.window = None
        self.clock = None
        if window:
            self.window = self.pg.display.set_mode((self.painter.w, self.painter.h))
            self.pg.display.set_caption("DroneDash")
            self.clock = self.pg.time.Clock()
        self.surface = self.pg.Surface((self.painter.w, self.painter.h))

    def rgb_array(self, sim) -> np.ndarray:
        self.painter.draw(self.surface, sim)
        return np.transpose(self.pg.surfarray.array3d(self.surface), (1, 0, 2)).copy()

    def show(self, sim) -> None:
        self.pg.event.pump()
        self.painter.draw(self.window, sim)
        self.pg.display.flip()
        self.clock.tick(C.FPS)

    def close(self) -> None:
        if self.window is not None:
            self.pg.display.quit()
            self.window = None


def _is_macos() -> bool:
    import sys
    return sys.platform == "darwin"
