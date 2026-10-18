"""Arc: the System 1 pilot (SPEC §13), ported from
``reference/arc_prototype.py``.

Tokens: SELF, RAYS, up to 12 HAZARD, 5 ACTION, DECIDE; no position embeddings (a set).
An entity encoder (``enc_layers`` attention blocks) runs with a structured mask: context tokens
see valid context, each ACTION token sees context and itself (never another action), DECIDE sees
valid context, the available actions and itself. A choice head (``head_layers`` blocks) lets the
ACTION tokens and DECIDE compare. Heads:

* CHOICE  ``logits``        bilinear DECIDE x ACTION, unavailable actions = -inf
* NOUL    ``safe_logits``   per-action logit of "still flying after 1 s"
* SCORE   ``threat_logits`` threat level 0-4 from DECIDE
* ``decide_attn``           DECIDE's attention over the 12 hazard slots in the last encoder
                            layer, averaged over heads (padding slots get 0)
* ``value``                 critic from DECIDE (training only; not exported)

The static part of the attention mask is precomputed once; per call only the hazard mask and
``act_avail`` change.

``ttc_features`` (off by default): each hazard token also gets 5 numbers computed from what the
observation already holds (no new sensor, same inputs): time to contact (gap / closing speed,
capped at 3 s), the climb and the descent needed to pass over or under it, and the slack of a
plain ASCEND / DESCEND started now (what 3 m/s covers in that time minus what is needed). So
"is there time for a slow move, or does this need a burst?" is a direct input.

``haz_cols`` (SPEC v2.3: 13): hazard rows carry the tracker's 5 values (watched, seen low / high,
predicted low / high); the climb and descent in ``ttc_features`` are then measured to the predicted
band, the one the teacher plans with. ``ffn_mult``: width of each block's feed-forward layer as a
multiple of d (2 before v2.3).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from drone_env import constants as C

N_HAZ, D_SELF, D_RAYS, D_HAZ, D_ADESC, N_ACT = 12, 11, 13, 8, 3, 5    # D_HAZ: rows before v2.3
N_TYPES = C.N_HAZARD_TYPES       # hazard type ids 0-12 (SPEC v2.0; 8 before)
D_TTC = 5                        # ttc_features per hazard token
TTC_CAP_S = 3.0
SLOW_RAMP_M = C.V_ASCEND ** 2 / (2 * C.A_MAX)   # distance lost while ASCEND/DESCEND reach 3 m/s
CTX = 2 + N_HAZ                  # SELF, RAYS, hazards
T_ALL = CTX + N_ACT + 1          # + actions + DECIDE
HAZ0 = 2                         # first hazard token
ACT0 = CTX                       # first action token
DEC = T_ALL - 1


class Attn(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x, allow, need_weights: bool = False):     # allow: [B, T, T] bool
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        sc = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        sc = sc.masked_fill(~allow[:, None], float("-inf"))
        p = sc.softmax(-1)                                          # [B, h, T, T]
        y = self.o((p @ v).transpose(1, 2).reshape(B, T, D))
        return (y, p) if need_weights else (y, None)


class Block(nn.Module):
    def __init__(self, d: int, h: int, ffn_mult: int = 2):
        super().__init__()
        self.n1, self.n2, self.a = nn.LayerNorm(d), nn.LayerNorm(d), Attn(d, h)
        self.m = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x, allow, need_weights: bool = False):
        y, p = self.a(self.n1(x), allow, need_weights)
        x = x + y
        return x + self.m(self.n2(x)), p


def _mlp(i: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, d))


def hazard_physics(self_f: torch.Tensor, haz: torch.Tensor) -> torch.Tensor:
    """[B, 12, 8 or 13] hazard tokens (+ SELF for the drone height) -> [B, 12, 5]: time to contact
    / 3 s, climb needed / 10 m, descent needed / 10 m, and ASCEND / DESCEND slack (m covered at 3 m/s
    in that time minus the climb / descent needed, / 5 m, clipped to +-1). Heights: the predicted
    band (columns 11-12) in v2.3 rows, else y_bottom / y_top."""
    lo_c, hi_c = (11, 12) if haz.shape[-1] >= 13 else (1, 2)
    nr = C.NORM_RANGE
    gap = haz[..., 0] * nr                                   # nose to leading edge, m
    closing = (-haz[..., 4] * nr).clamp_min(1.0)             # m/s; stationary hazards: drone speed
    ttc = (gap / closing).clamp(max=TTC_CAP_S)
    y = self_f[:, None, 0] * 10.0
    up = (haz[..., hi_c] * 10.0 - (y - C.HALF_H)).clamp_min(0.0)
    down = ((y + C.HALF_H) - haz[..., lo_c] * 10.0).clamp_min(0.0)
    reach = (C.V_ASCEND * ttc - SLOW_RAMP_M).clamp_min(0.0)
    return torch.stack([ttc / TTC_CAP_S, up / 10.0, down / 10.0, ((reach - up) / 5.0).clamp(-1.0, 1.0),
                        ((reach - down) / 5.0).clamp(-1.0, 1.0)], -1)


def _static_masks():
    """Structural attention pattern S (rows = queries) and the identity, for T_ALL tokens."""
    S = torch.zeros(T_ALL, T_ALL, dtype=torch.bool)
    S[:CTX, :CTX] = True                       # context sees context
    S[ACT0:ACT0 + N_ACT, :CTX] = True          # each action sees context ...
    S[DEC, :] = True                           # DECIDE sees everything (filtered by validity)
    eye = torch.eye(T_ALL, dtype=torch.bool)   # ... and itself; padded rows see themselves (no NaN)
    return S, eye


class ArcNet(nn.Module):
    def __init__(self, d: int = 64, heads: int = 4, enc_layers: int = 2, head_layers: int = 1,
                 ttc_features: bool = False, haz_cols: int = D_HAZ, ffn_mult: int = 2):
        super().__init__()
        self.cfg = {"d": d, "heads": heads, "enc_layers": enc_layers, "head_layers": head_layers}
        if ttc_features:
            self.cfg["ttc_features"] = True
        if haz_cols != D_HAZ:
            self.cfg["haz_cols"] = int(haz_cols)
        if ffn_mult != 2:
            self.cfg["ffn_mult"] = int(ffn_mult)
        self.haz_cols = int(haz_cols)
        self.d = d
        self.ttc_features = bool(ttc_features)
        self.self_enc, self.ray_enc = _mlp(D_SELF, d), _mlp(D_RAYS, d)
        self.haz_enc = _mlp(haz_cols + (D_TTC if ttc_features else 0), d)
        self.haz_type = nn.Embedding(N_TYPES, d)
        self.act_id, self.act_desc = nn.Embedding(N_ACT, d), _mlp(D_ADESC, d)
        self.decide = nn.Parameter(torch.randn(d) * 0.02)
        self.enc = nn.ModuleList([Block(d, heads, ffn_mult) for _ in range(enc_layers)])
        self.head = nn.ModuleList([Block(d, heads, ffn_mult) for _ in range(head_layers)])
        self.norm = nn.LayerNorm(d)
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, d)   # bilinear DECIDE x ACTION (CHOICE)
        self.safe = nn.Linear(d, 1)                                    # per-action P(safe for 1 s) (NOUL)
        self.threat = nn.Linear(d, 5)                                  # threat 0-4 (SCORE)
        self.value = nn.Linear(d, 1)                                   # critic, training only
        S, eye = _static_masks()
        self.register_buffer("mask_struct", S, persistent=False)
        self.register_buffer("mask_eye", eye, persistent=False)
        self.register_buffer("head_eye", torch.eye(N_ACT + 1, dtype=torch.bool), persistent=False)

    def forward(self, self_f, rays, haz, haz_type, haz_mask, act_ids, act_desc, act_avail):
        """Returns (logits, safe_logits, threat_logits, decide_attn, value)."""
        B = haz.shape[0]
        if self.ttc_features:
            haz = torch.cat([haz, hazard_physics(self_f, haz)], -1)
        ctx = torch.cat([self.self_enc(self_f)[:, None], self.ray_enc(rays)[:, None],
                         self.haz_enc(haz) + self.haz_type(haz_type)], 1)          # [B, CTX, d]
        acts = self.act_id(act_ids) + self.act_desc(act_desc)                       # [B, A, d]
        x = torch.cat([ctx, acts, self.decide.expand(B, 1, self.d)], 1)             # [B, T, d]
        ones2 = torch.ones(B, 2, dtype=torch.bool, device=haz.device)
        ones1 = torch.ones(B, 1, dtype=torch.bool, device=haz.device)
        col_ok = torch.cat([ones2, haz_mask, act_avail, ones1], 1)                   # [B, T]
        allow = (self.mask_struct[None] & col_ok[:, None, :]) | self.mask_eye[None]
        p_last = None
        n_enc = len(self.enc)
        for i, blk in enumerate(self.enc):
            x, p = blk(x, allow, need_weights=(i == n_enc - 1))
            if p is not None:
                p_last = p
        h = x[:, ACT0:]                                                              # [actions..., DECIDE]
        hk = torch.cat([act_avail, ones1], 1)
        hallow = hk[:, None, :] | self.head_eye[None]
        for blk in self.head:
            h, _ = blk(h, hallow)
        h = self.norm(h)
        a, dec = h[:, :N_ACT], h[:, -1]
        logits = torch.einsum("bd,bad->ba", self.q_proj(dec), self.k_proj(a)) / math.sqrt(self.d)
        logits = logits.masked_fill(~act_avail, float("-inf"))
        decide_attn = p_last[:, :, DEC, HAZ0:HAZ0 + N_HAZ].mean(1)                     # [B, 12]
        return logits, self.safe(a).squeeze(-1), self.threat(dec), decide_attn, self.value(dec).squeeze(-1)


def arc_from_config(cfg: dict) -> ArcNet:
    return ArcNet(d=int(cfg.get("d", 64)), heads=int(cfg.get("heads", 4)),
                           enc_layers=int(cfg.get("enc_layers", 2)), head_layers=int(cfg.get("head_layers", 1)),
                           ttc_features=bool(cfg.get("ttc_features", False)),
                           haz_cols=int(cfg.get("haz_cols", D_HAZ)), ffn_mult=int(cfg.get("ffn_mult", 2)))
