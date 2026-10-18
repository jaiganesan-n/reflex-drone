"""Optional practice: PPO fine-tuning of Arc in the real env (CleanRL style, one file).

    uv run python -m train.ppo --config configs/train/ppo.yaml --init runs/arc_B/best.pt --run arc_ppo [--time-budget-min 180]

* 8 envs in ``gymnasium.vector.AsyncVectorEnv`` (spawn). Each env resets itself at the end of
  an episode (fresh TRAIN seed, current curriculum level) and reports the episode in ``info``.
* The actor starts from ``--init``. For the first 50k env steps only the value head trains
  (actor frozen), then PPO: lr 1e-4, clip 0.1, entropy 0.005, 1,024 steps per env per update.
* The supervised contract losses (KL + safe BCE + threat CE) stay on at weight 0.1, on
  teacher frames replayed from ``replay_data`` (data/teacher).
* Curriculum L1 -> L5: move up when the rolling mean distance reaches 80% of that level's
  planner mean (VAL planner cache). Or ``level_mix`` ({level: weight}): every episode draws its
  level, no curriculum steps (the rolling number is then the mean distance ratio).
* Optional training-only shaping, added to the game reward (SPEC §6, unchanged) inside the
  training envs: ``clearance_bonus`` * min(clearance, ``clearance_cap_m``) / cap per frame
  survived (clearance = metres to the nearest hazard, ground or ceiling), ``-change_penalty`` per
  action change, ``-target_change_penalty`` per m/s of change in the target vertical speed (so
  ASCEND -> HOLD -> DESCEND costs what ASCEND -> DESCEND does), ``-vertical_speed_penalty`` *
  |vy| / 3 m/s per frame survived, ``-burst_penalty`` per JUMP or DUCK started, and
  ``-needless_burst_penalty`` more when the teacher's safety check (``teacher.planner.plan``, 1 s)
  finds HOLD, ASCEND or DESCEND safe at that moment. ``-slow_burst_penalty`` more per burst at
  8 m/s, fading linearly to 0 at 16 m/s (a burst at low speed is rarely needed). All 0 by default.
  The teacher runs only in training, only on frames where a burst starts; ``teacher_config`` picks
  its settings file (default configs/train/teacher.yaml) and ``needless_safe_frames`` > 0 replaces
  its safe_frames for this check (e.g. 45 = the whole 1.5 s plan: needless only if a slower move
  gets through the whole plan).
  ``path_bonus`` * min(path_time, ``path_horizon_s``) / ``path_horizon_s`` per frame survived, where
  path_time (``Simulator.path_time``, margin 0.3 m) is how long flying level at the current height
  stays clear: practice pays for moving out of an obstacle's way early, not at the last moment.
* ``policy_temperature``: PPO's policy is softmax(logits / temperature) when it samples and when
  it updates (default 1.0). An imitation model's raw logits copy the teacher's soft labels, so
  sampling them at 1.0 flies far noisier than the greedy pilot the app uses; the calibrated T
  (train.calibrate) keeps practice close to play. The aux loss and the exported model still use
  the raw logits.
* ``aux_mask_needless_bursts``: on replayed teacher frames where HOLD, ASCEND or DESCEND is safe
  (the stored ``safe`` labels), the aux KL target drops the teacher's JUMP/DUCK probability
  (renormalized), so the anchor stops pulling needless bursts back in.
* ``ref_kl_coef`` > 0: + coef * KL(policy || starting policy) on the states PPO itself flies, both at
  ``policy_temperature`` (the starting policy is ``--init``, frozen). The aux anchor's soft teacher
  labels barely tell "start climbing now" from "hold a little longer", so on its own it never kept
  the start's timing; this anchor does, and PPO moves away from it only where the reward says so.
* best.pt = best online ratio on 10 VAL seeds of ``gate_level`` (or the mean over
  ``select_levels``), checked every ``eval_every`` updates and once for the initial model; with
  ``select_after`` > 0 only updates from then on can become best.pt (the final model if none did).
  ``snapshot_every`` > 0 also keeps ``update_<n>.pt`` every n updates. At the end best.pt is flown
  on all 50 ``gate_level`` VAL seeds -> summary.json val_ratio.
  ``--time-budget-min`` stops cleanly after the current update.

torch is imported inside functions so the env worker processes (spawn) stay light.
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from drone_env import constants as C
from drone_env.env import DroneDashEnv, observation_space
from drone_env.seeds import SPLITS

PPO_DEFAULTS = {
    "num_envs": 8, "steps_per_env": 1024, "lr": 1.0e-4, "clip": 0.1, "ent_coef": 0.005, "vf_coef": 0.5,
    "gamma": 0.99, "gae_lambda": 0.95, "update_epochs": 4, "minibatch_size": 1024, "max_grad_norm": 0.5,
    "critic_warmup_steps": 50000, "aux_weight": 0.1, "aux_batch": 256, "replay_data": "data/teacher",
    "curriculum": ["L1", "L2", "L3", "L4", "L5"], "advance_frac": 0.8, "rolling_episodes": 20,
    "episode_frame_cap": 6000, "seeds": "train", "eval_every": 10, "gate_level": "L5", "online_seeds": 10,
    "eval_frame_cap": 6000, "final_val_seeds": 50, "planner_cache_dir": "reports", "workers": 6,
    "time_budget_min": 180, "total_steps": 0, "seed": 0, "threads": 0, "device": "cpu",
    "level_mix": None, "select_levels": None, "policy_temperature": 1.0,
    "select_after": 0, "snapshot_every": 0, "aux_mask_needless_bursts": False,
    "clearance_bonus": 0.0, "clearance_cap_m": 2.0, "change_penalty": 0.0, "burst_penalty": 0.0,
    "target_change_penalty": 0.0, "vertical_speed_penalty": 0.0, "needless_burst_penalty": 0.0,
    "slow_burst_penalty": 0.0, "needless_safe_frames": 0, "teacher_config": None,
    "path_bonus": 0.0, "path_horizon_s": 1.2, "ref_kl_coef": 0.0,
}
SHAPING_KEYS = ("clearance_bonus", "clearance_cap_m", "change_penalty", "burst_penalty", "target_change_penalty",
                "vertical_speed_penalty", "needless_burst_penalty", "slow_burst_penalty", "needless_safe_frames",
                "path_bonus", "path_horizon_s")


class CurriculumEnv(gym.Env):
    """DroneDash that resets itself: random seed from ``split``, level set by ``set_level`` (or
    drawn per episode from ``level_mix``), reward plus the optional training-only ``shaping``.

    Never returns terminated/truncated (so the vector env never auto-resets); instead ``info``
    carries ``ep_done``, ``ep_crash``, ``ep_distance``, ``ep_frames``, ``ep_level``, and the
    episode's ``ep_clearance`` (mean, m; only with a clearance bonus), ``ep_changes``, ``ep_bursts``,
    ``ep_needless`` (only with a needless-burst penalty) and ``ep_holds`` (HOLD frames) on the step
    that ended an episode, and the returned observation is the first one of the next episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, index: int = 0, seed: int = 0, split: str = "train", frame_cap: int = 6000,
                 level: str = "L1", level_mix: dict | None = None, shaping: dict | None = None,
                 teacher_config: str | None = None):
        self.observation_space = observation_space()
        self.action_space = gym.spaces.Discrete(C.N_ACTIONS)
        self.rng = np.random.default_rng([seed, index, 0xC0FFEE])
        self.split = SPLITS[split]
        self.frame_cap = frame_cap
        self.level = level
        self.next_level = level
        self.mix = None
        if level_mix:
            w = np.array([float(v) for v in level_mix.values()])
            self.mix = (list(level_mix), w / w.sum())
        sh = {k: float(PPO_DEFAULTS[k]) for k in SHAPING_KEYS}
        sh.update({k: float(v) for k, v in (shaping or {}).items()})
        self.shaping = sh
        self.teacher_path = teacher_config
        self.teacher_cfg = None
        self.env = None

    def set_level(self, level: str) -> bool:
        self.next_level = level
        return True

    def _new_episode(self):
        if self.mix:
            names, p = self.mix
            self.level = names[int(self.rng.choice(len(names), p=p))]
        else:
            self.level = self.next_level
        self.env = DroneDashEnv(level=self.level, render_mode=None, max_frames=self.frame_cap)
        s = int(self.rng.integers(self.split.start, self.split.stop))
        obs, _ = self.env.reset(seed=s)
        self.ep_clear, self.ep_changes, self.ep_bursts, self.ep_needless, self.ep_holds = 0.0, 0, 0, 0, 0
        return obs

    @staticmethod
    def _blank():
        return {"ep_done": False, "ep_crash": False, "ep_distance": 0.0, "ep_frames": 0, "ep_level": -1,
                "ep_clearance": 0.0, "ep_changes": 0, "ep_bursts": 0, "ep_needless": 0, "ep_holds": 0}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._new_episode(), self._blank()

    def _needless_burst(self, action: int) -> bool:
        """Is ``action`` a burst that would start now while HOLD, ASCEND or DESCEND is safe?"""
        sim = self.env.sim
        if action not in (C.JUMP, C.DUCK) or action not in sim.available_actions():
            return False
        from dataclasses import replace
        from teacher.planner import load_teacher_config, plan
        if self.teacher_cfg is None:
            self.teacher_cfg = load_teacher_config(self.teacher_path)
            if self.shaping["needless_safe_frames"] > 0:
                self.teacher_cfg = replace(self.teacher_cfg, safe_frames=int(self.shaping["needless_safe_frames"]))
        return bool(plan(sim, self.teacher_cfg).safe[:C.JUMP].any())

    def _shape(self, prev_action: int, prev_vt: float, crashed: bool, needless: bool, speed: float = 16.0) -> float:
        sim, sh = self.env.sim, self.shaping
        r = 0.0
        if sim.last_action != prev_action:
            self.ep_changes += 1
            r -= sh["change_penalty"]
        r -= sh["target_change_penalty"] * abs(sim.v_target - prev_vt)
        if sim.last_action in (C.JUMP, C.DUCK):       # executed only on the frame a burst starts
            self.ep_bursts += 1
            r -= sh["burst_penalty"] + sh["slow_burst_penalty"] * min(max((16.0 - speed) / 8.0, 0.0), 1.0)
            if needless:
                self.ep_needless += 1
                r -= sh["needless_burst_penalty"]
        elif sim.last_action == C.HOLD and sim.burst == 0:
            self.ep_holds += 1
        if not crashed:
            r -= sh["vertical_speed_penalty"] * abs(sim.vy) / 3.0
        if sh["clearance_bonus"] and not crashed:
            cl = sim.clearance()
            self.ep_clear += cl
            r += sh["clearance_bonus"] * min(cl, sh["clearance_cap_m"]) / sh["clearance_cap_m"]
        if sh["path_bonus"] and not crashed:
            H = sh["path_horizon_s"]
            r += sh["path_bonus"] * sim.path_time(H, 0.3) / H
        return r

    def step(self, action):
        sim = self.env.sim
        prev, prev_vt, speed = sim.last_action, sim.v_target, sim.speed
        needless = self._needless_burst(int(action)) if self.shaping["needless_burst_penalty"] else False
        obs, r, term, trunc, info = self.env.step(int(action))
        r += self._shape(prev, prev_vt, term, needless, speed)
        out = self._blank()
        if term or trunc:
            n = max(int(info["frame"]), 1)
            out.update(ep_done=True, ep_crash=bool(term), ep_distance=float(info["distance_m"]),
                       ep_frames=int(info["frame"]), ep_level=C.LEVEL_CODES[self.level],
                       ep_clearance=self.ep_clear / n, ep_changes=self.ep_changes, ep_bursts=self.ep_bursts,
                       ep_needless=self.ep_needless, ep_holds=self.ep_holds)
            obs = self._new_episode()
        return obs, float(r), False, False, out


def mask_needless_bursts(p_teacher, safe):
    """Teacher targets with no JUMP/DUCK mass on frames where HOLD, ASCEND or DESCEND is safe (renormalized)."""
    calm = (safe[:, :C.JUMP] > 0.5).any(-1, keepdim=True)
    q = p_teacher.clone()
    q[:, C.JUMP:] = q[:, C.JUMP:].masked_fill(calm, 0.0)
    return q / q.sum(-1, keepdim=True).clamp_min(1e-12)


def ref_kl(logits, ref_logp, avail, temperature: float = 1.0):
    """Mean KL(softmax(logits / T) || exp(ref_logp)) over the available actions (masked ones add 0
    and get no gradient). ``ref_logp`` is the frozen starting policy's log_softmax(logits / T)."""
    import torch
    logp = torch.log_softmax(logits / temperature, -1).masked_fill(~avail, 0.0)
    p = logp.exp() * avail
    return (p * (logp - ref_logp.masked_fill(~avail, 0.0))).sum(-1).mean()


def _make_env(index, seed, split, frame_cap, level, level_mix=None, shaping=None, teacher_config=None):
    return CurriculumEnv(index, seed, split, frame_cap, level, level_mix, shaping, teacher_config)


def _obs_to_batch(obs: dict, act_desc, device):
    import torch
    B = obs["self"].shape[0]
    return {
        "self": torch.as_tensor(obs["self"], dtype=torch.float32, device=device),
        "rays": torch.as_tensor(obs["rays"], dtype=torch.float32, device=device),
        "hazards": torch.as_tensor(obs["hazards"], dtype=torch.float32, device=device),
        "hazard_type": torch.as_tensor(obs["hazard_type"], dtype=torch.int64, device=device),
        "hazard_mask": torch.as_tensor(obs["hazard_mask"], device=device).bool(),
        "act_avail": torch.as_tensor(obs["action_mask"], device=device).bool(),
        "act_ids": torch.arange(5, device=device).expand(B, 5),
        "act_desc": act_desc.to(device).expand(B, 5, 3),
    }


def planner_means(levels, cfg) -> dict:
    from drone_env.seeds import seeds_for
    from eval.planner_cache import ensure
    out = {}
    for lv in levels:
        seeds = seeds_for("val", lv)[: int(cfg["online_seeds"])]
        e = ensure(cfg["planner_cache_dir"], "val", lv, seeds, int(cfg["eval_frame_cap"]), int(cfg["workers"]))
        out[lv] = float(np.mean([e[s]["distance_m"] for s in seeds]))
    return out


def train_ppo(cfg: dict, init: str, run_name: str) -> dict:
    import torch
    from data.dataset import FrameData
    from drone_env.env import action_descriptors
    from models import load_checkpoint, policy_outputs, save_checkpoint
    from train.common import Run, as_list, contract_loss, final_val, online_eval, set_threads

    torch.manual_seed(int(cfg["seed"]))
    set_threads(int(cfg["threads"]))
    run = Run(run_name, {**cfg, "init": init})
    model, ck = load_checkpoint(init)
    if ck["arch"] != "arc":
        raise SystemExit("PPO fine-tunes an Arc checkpoint")
    device = torch.device("cpu" if cfg["device"] in ("auto", "cpu") else cfg["device"])
    model.to(device).train()
    ref = None
    if float(cfg["ref_kl_coef"]) > 0:
        ref, _ = load_checkpoint(init)
        ref.to(device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        print(f"[ppo] anchor: {cfg['ref_kl_coef']} x KL(policy || {init}) on its own states", flush=True)
    act_desc = torch.from_numpy(action_descriptors())
    replay = FrameData(as_list(cfg["replay_data"])) if float(cfg["aux_weight"]) > 0 else None
    mix = dict(cfg["level_mix"]) if cfg["level_mix"] else None
    levels = list(mix) if mix else list(cfg["curriculum"])
    select = list(cfg["select_levels"] or [cfg["gate_level"]])
    pmeans = planner_means(list(dict.fromkeys(levels + select)), cfg)
    print(f"[ppo] planner means (VAL, cap {cfg['eval_frame_cap']}): "
          + ", ".join(f"{k} {v:.0f} m" for k, v in pmeans.items()), flush=True)
    shaping = {k: float(cfg[k]) for k in SHAPING_KEYS}
    if mix or any(shaping[k] for k in SHAPING_KEYS if k not in ("clearance_cap_m", "needless_safe_frames", "path_horizon_s")):
        print(f"[ppo] level mix {mix}; training-only shaping {shaping}", flush=True)
    if float(cfg["policy_temperature"]) != 1.0:
        print(f"[ppo] policy temperature {float(cfg['policy_temperature'])}", flush=True)

    N, S = int(cfg["num_envs"]), int(cfg["steps_per_env"])
    pi_T = float(cfg["policy_temperature"])
    lvl_i = 0
    fns = [functools.partial(_make_env, i, int(cfg["seed"]), cfg["seeds"], int(cfg["episode_frame_cap"]), levels[0],
                             mix, shaping, cfg["teacher_config"])
           for i in range(N)]
    envs = gym.vector.AsyncVectorEnv(fns, shared_memory=True, context="spawn")
    obs, _ = envs.reset(seed=int(cfg["seed"]))

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), eps=1e-5, weight_decay=0.0)
    budget_s = float(cfg["time_budget_min"]) * 60.0
    t_start = time.time()
    global_step, update = 0, 0
    rolling = collections.deque(maxlen=int(cfg["rolling_episodes"]))
    ep_stats = collections.deque(maxlen=int(cfg["rolling_episodes"]))   # (clearance m, changes/s, bursts/min, needless, bursts, HOLD share)
    best_ratio, best_update = -math.inf, None

    def evaluate(tag):
        nonlocal best_ratio, best_update
        per = {lv: online_eval(model, "arc", lv, int(cfg["online_seeds"]), int(cfg["eval_frame_cap"]),
                               cfg["planner_cache_dir"], int(cfg["workers"]), device)["ratio"] for lv in select}
        score = float(np.mean(list(per.values())))
        if update >= int(cfg["select_after"]) and score > best_ratio:
            best_ratio, best_update = score, update
            save_checkpoint(run.path("best.pt"), model, "arc", stage="ppo" if update else ck.get("stage", "ppo"),
                            run=run_name, update=update, global_step=global_step, config=cfg, online_ratio=score,
                            online_ratios=per)
        what = select[0] if len(select) == 1 else "mean " + " ".join(f"{k} {v:.3f}" for k, v in per.items()) + " ->"
        print(f"[ppo] eval ({tag}) update {update}: online {what} ratio {score:.3f} "
              f"(best {best_ratio:.3f} @ update {best_update})", flush=True)
        return {"ratio": score, "ratios": per}

    evaluate("init")
    keys = ("self", "rays", "hazards", "hazard_type", "hazard_mask", "act_avail")
    while True:
        elapsed = time.time() - t_start
        if elapsed >= budget_s or (cfg["total_steps"] and global_step >= int(cfg["total_steps"])):
            break
        t_upd = time.time()
        buf = {k: [] for k in keys}
        acts, logps, vals, rews, dones = [], [], [], [], []
        model.eval()
        for _ in range(S):
            b = _obs_to_batch(obs, act_desc, device)
            with torch.no_grad():
                out = policy_outputs(model, "arc", b)
                dist = torch.distributions.Categorical(logits=out["logits"] / pi_T)
                a = dist.sample()
            for k in keys:
                buf[k].append(b[k])
            acts.append(a)
            logps.append(dist.log_prob(a))
            vals.append(out["value"])
            obs, r, _t, _tr, info = envs.step(a.cpu().numpy())
            done = np.asarray(info["ep_done"], bool)
            rews.append(torch.as_tensor(r, dtype=torch.float32, device=device))
            dones.append(torch.as_tensor(done, dtype=torch.float32, device=device))
            for i in np.flatnonzero(done):
                lv = C.LEVELS[int(info["ep_level"][i])]
                if mix:
                    rolling.append(float(info["ep_distance"][i]) / pmeans[lv])
                elif lv == levels[lvl_i]:
                    rolling.append(float(info["ep_distance"][i]))
                secs = max(int(info["ep_frames"][i]), 1) / C.FPS
                ep_stats.append((float(info["ep_clearance"][i]), int(info["ep_changes"][i]) / secs,
                                 60.0 * int(info["ep_bursts"][i]) / secs, int(info["ep_needless"][i]),
                                 int(info["ep_bursts"][i]), int(info["ep_holds"][i]) / max(int(info["ep_frames"][i]), 1)))
            global_step += N
        rollout_s = time.time() - t_upd
        with torch.no_grad():
            nv = policy_outputs(model, "arc", _obs_to_batch(obs, act_desc, device))["value"]
        R = torch.stack(rews)
        D = torch.stack(dones)
        V = torch.stack(vals)
        adv = torch.zeros_like(R)
        last = torch.zeros(N, device=device)
        g, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
        for t in reversed(range(S)):
            nxt = nv if t == S - 1 else V[t + 1]
            nonterm = 1.0 - D[t]
            delta = R[t] + g * nxt * nonterm - V[t]
            last = delta + g * lam * nonterm * last
            adv[t] = last
        ret = adv + V
        flat = {k: torch.cat(buf[k]) for k in keys}
        A = torch.cat(acts)
        LP = torch.cat(logps)
        ADV = adv.reshape(-1)
        RET = ret.reshape(-1)
        warm = global_step <= int(cfg["critic_warmup_steps"])
        n = A.shape[0]
        mb = int(cfg["minibatch_size"])
        REF = None
        if ref is not None and not warm:
            with torch.no_grad():
                REF = torch.cat([torch.log_softmax(policy_outputs(ref, "arc", {k: flat[k][s0:s0 + mb] for k in keys})
                                                   ["logits"] / pi_T, -1) for s0 in range(0, n, mb)])
        for name, p in model.named_parameters():
            p.requires_grad_(not warm or name.startswith("value."))
        model.train()
        stats = collections.defaultdict(float)
        nmb = 0
        for _ in range(int(cfg["update_epochs"])):
            perm = torch.randperm(n, device=device)
            for s0 in range(0, n, mb):
                idx = perm[s0:s0 + mb]
                b = {k: flat[k][idx] for k in keys}
                B = len(idx)
                b["act_ids"] = torch.arange(5, device=device).expand(B, 5)
                b["act_desc"] = act_desc.to(device).expand(B, 5, 3)
                out = policy_outputs(model, "arc", b)
                v_loss = 0.5 * ((out["value"] - RET[idx]) ** 2).mean()
                if warm:
                    loss = float(cfg["vf_coef"]) * v_loss
                else:
                    dist = torch.distributions.Categorical(logits=out["logits"] / pi_T)
                    newlp = dist.log_prob(A[idx])
                    ratio = (newlp - LP[idx]).exp()
                    ad = ADV[idx]
                    ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                    clip = float(cfg["clip"])
                    pg = torch.max(-ad * ratio, -ad * ratio.clamp(1 - clip, 1 + clip)).mean()
                    ent = dist.entropy().mean()
                    loss = pg - float(cfg["ent_coef"]) * ent + float(cfg["vf_coef"]) * v_loss
                    stats["pg"] += float(pg)
                    stats["entropy"] += float(ent)
                    stats["clipfrac"] += float(((ratio - 1).abs() > clip).float().mean())
                    if REF is not None:
                        rkl = ref_kl(out["logits"], REF[idx], b["act_avail"], pi_T)
                        loss = loss + float(cfg["ref_kl_coef"]) * rkl
                        stats["ref_kl"] += float(rkl)
                    if replay is not None:
                        ridx = replay.train_idx[torch.randint(len(replay.train_idx), (int(cfg["aux_batch"]),))]
                        rb = replay.batch(ridx, device)
                        if cfg["aux_mask_needless_bursts"]:
                            rb["p_teacher"] = mask_needless_bursts(rb["p_teacher"], rb["safe"])
                        aux, _ = contract_loss(policy_outputs(model, "arc", rb), rb, "arc")
                        loss = loss + float(cfg["aux_weight"]) * aux
                        stats["aux"] += float(aux)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
                opt.step()
                stats["v_loss"] += float(v_loss)
                nmb += 1
        for p in model.parameters():
            p.requires_grad_(True)
        update += 1
        upd_s = time.time() - t_upd
        roll = float(np.mean(rolling)) if rolling else 0.0
        es = np.mean(ep_stats, axis=0) if ep_stats else np.zeros(6)
        needless_share = (sum(x[3] for x in ep_stats) / max(sum(x[4] for x in ep_stats), 1)) if ep_stats else 0.0
        row = {"update": update, "global_step": global_step, "level": "mix" if mix else levels[lvl_i], "warmup": warm,
               "episodes_in_window": len(rolling), "ep_clearance_m": float(es[0]), "ep_changes_per_s": float(es[1]),
               "ep_bursts_per_min": float(es[2]), "ep_needless_share": float(needless_share), "ep_hold_share": float(es[5]),
               "steps_per_s": round(N * S / max(upd_s, 1e-9), 1), "rollout_steps_per_s": round(N * S / max(rollout_s, 1e-9), 1),
               "update_seconds": round(upd_s, 1), **{k: v / max(nmb, 1) for k, v in stats.items()}}
        if mix:
            row["rolling_ratio"] = roll
            where = f"mix rolling ratio {roll:.2f}"
        else:
            row.update(rolling_distance_m=roll, planner_mean_m=pmeans[levels[lvl_i]])
            where = f"{levels[lvl_i]} rolling {roll:.0f} m / planner {pmeans[levels[lvl_i]]:.0f} m"
        print(f"[ppo] update {update} step {global_step:,} {'(critic warm-up) ' if warm else ''}{where}  "
              f"clear {es[0]:.2f} m, {es[1]:.2f} changes/s, {es[2]:.1f} bursts/min"
              + (f" ({needless_share:.0%} needless)" if shaping["needless_burst_penalty"] else "")
              + f", HOLD {es[5]:.0%}"
              + (f", KL to start {row['ref_kl']:.3f}" if "ref_kl" in row else "")
              + f"  {row['steps_per_s']:.0f} steps/s",
              flush=True)
        # curriculum
        if (not mix and len(rolling) >= max(4, int(cfg["rolling_episodes"]) // 2) and lvl_i < len(levels) - 1
                and roll >= float(cfg["advance_frac"]) * pmeans[levels[lvl_i]]):
            lvl_i += 1
            envs.call("set_level", levels[lvl_i])
            rolling.clear()
            print(f"[ppo] curriculum -> {levels[lvl_i]}", flush=True)
            row["advanced_to"] = levels[lvl_i]
        if int(cfg["snapshot_every"]) and update % int(cfg["snapshot_every"]) == 0:
            save_checkpoint(run.path(f"update_{update}.pt"), model, "arc", stage="ppo", run=run_name, update=update,
                            global_step=global_step, config=cfg)
        if update % max(1, int(cfg["eval_every"])) == 0:
            on = evaluate("periodic")
            row["online_ratio"] = on["ratio"]
            row["online_ratios"] = on["ratios"]
        run.log(row)
    envs.close()
    save_checkpoint(run.path("last.pt"), model, "arc", stage="ppo", run=run_name, update=update,
                    global_step=global_step, config=cfg)
    if update and update % max(1, int(cfg["eval_every"])) != 0:
        evaluate("final")
    if best_update is None:          # no update reached select_after: best.pt = the final model
        best_update = update
        save_checkpoint(run.path("best.pt"), model, "arc", stage="ppo", run=run_name, update=update,
                        global_step=global_step, config=cfg)
    wall = time.time() - t_start
    summary = {"run": run_name, "arch": "arc", "stage": "ppo", "init": init, "updates": update,
               "global_steps": global_step, "best_update": best_update, "best_online_ratio": best_ratio,
               "final_level": "mix" if mix else levels[lvl_i], "seconds": round(wall, 1),
               "steps_per_s": round(global_step / max(wall, 1e-9), 1)}
    summary.update(final_val(run.path("best.pt"), cfg["gate_level"], int(cfg["final_val_seeds"]),
                             int(cfg["eval_frame_cap"]), cfg["planner_cache_dir"], int(cfg["workers"])))
    run.write_summary(summary)
    print(f"[ppo] done: {update} updates, {global_step:,} steps in {wall / 60:.1f} min; best.pt (update {best_update}) "
          f"val_ratio {summary['val_ratio']:.3f} on {summary['val_seeds']} VAL seeds of {summary['val_level']}", flush=True)
    return summary


def main(argv=None) -> int:
    from train.common import load_config
    ap = argparse.ArgumentParser(prog="python -m train.ppo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train/ppo.yaml")
    ap.add_argument("--init", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--time-budget-min", type=float, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, PPO_DEFAULTS)
    if args.time_budget_min is not None:
        cfg["time_budget_min"] = args.time_budget_min
    train_ppo(cfg, args.init, args.run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
