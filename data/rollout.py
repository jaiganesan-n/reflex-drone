"""One labelled episode: somebody flies, the teacher labels every frame (replan 1).

Used by data.collect (teacher flies, with random recovery bursts), data.dagger (student mixed
with the teacher at beta) and train.calibrate (student flies). Runs inside worker processes, so
everything here is a plain function of a picklable task dict.
"""
from __future__ import annotations

import numpy as np

from drone_env import constants as C
from drone_env.env import DroneDashEnv
from teacher.planner import TeacherConfig, plan

HOLD = 0


def _empty(n_cap: int) -> dict:
    return {
        "self": np.zeros((n_cap, 11), np.float32),
        "rays": np.zeros((n_cap, 13), np.float16),
        "hazards": np.zeros((n_cap, 12, C.HAZ_COLS), np.float16),
        "hazard_type": np.zeros((n_cap, 12), np.int8),
        "hazard_mask": np.zeros((n_cap, 12), bool),
        "action_mask": np.zeros((n_cap, 5), bool),
        "teacher_scores": np.zeros((n_cap, 5), np.float32),
        "safe": np.zeros((n_cap, 5), bool),
        "threat": np.zeros(n_cap, np.int8),
        "action": np.zeros(n_cap, np.int8),
        "student_action": np.full(n_cap, -1, np.int8),
        "student_conf": np.zeros(n_cap, np.float16),
        "reward": np.zeros(n_cap, np.float32),
        "done": np.zeros(n_cap, bool),
        "episode_id": np.zeros(n_cap, np.int32),
        "step": np.zeros(n_cap, np.int32),
        "seed": np.zeros(n_cap, np.int32),
        "level": np.zeros(n_cap, np.int8),
        "tags": np.zeros((n_cap, 3), bool),
    }


def run_episode(task: dict) -> dict:
    """Fly one episode and label every frame with the teacher.

    task keys: level, seed, episode_id, frames_cap, teacher (TeacherConfig dict),
    burst_prob (0 = none: per-frame chance of a 1-5 frame burst of random actions drawn from
    the available actions the teacher marks safe), and optionally student = {"model": ckpt path, "temperature": T,
    "beta": prob of executing the teacher's action instead of the student's}.
    Returns the SPEC §12 arrays for this episode plus "executed" (int8) and "outcome".
    """
    level, seed, cap = task["level"], int(task["seed"]), int(task["frames_cap"])
    cfg = TeacherConfig(**task["teacher"])
    burst_prob = float(task.get("burst_prob", 0.0))
    stu = task.get("student")
    policy = None
    beta = 1.0
    if stu is not None:
        from models.policy import TorchPolicy
        policy = TorchPolicy(stu["model"], temperature=float(stu.get("temperature", 1.0)))
        beta = float(stu.get("beta", 0.0))
    rng = np.random.default_rng([seed, 0x5EED])        # bursts and beta mixing only

    env = DroneDashEnv(level=level, render_mode=None, max_frames=cap)
    obs, info = env.reset(seed=seed)
    sim = env.sim
    if policy is not None:
        policy.reset(env.action_descriptors())
    out = _empty(cap)
    executed = np.zeros(cap, np.int8)
    logits = np.zeros((cap, 5), np.float32) if policy is not None else None
    burst_left = 0
    i = 0
    lev = C.LEVEL_CODES[level]
    while True:
        p = plan(sim, cfg)
        out["self"][i] = obs["self"]
        out["rays"][i] = obs["rays"]
        out["hazards"][i] = obs["hazards"]
        out["hazard_type"][i] = obs["hazard_type"]
        out["hazard_mask"][i] = obs["hazard_mask"] != 0
        out["action_mask"][i] = obs["action_mask"] != 0
        out["teacher_scores"][i] = p.scores
        out["safe"][i] = p.safe
        out["threat"][i] = p.threat
        out["action"][i] = p.action
        out["step"][i] = i
        out["tags"][i] = (sim.gust_active, sim.fog_active, sim.surge_active)
        a = p.action
        if policy is not None:
            s_act, probs = policy.act(obs)
            logits[i] = policy.last_logits
            out["student_action"][i] = s_act
            out["student_conf"][i] = float(probs.max())
            if rng.random() >= beta:
                a = s_act
        if burst_left > 0 or (burst_prob > 0.0 and rng.random() < burst_prob):
            if burst_left == 0:
                burst_left = int(rng.integers(1, 6))      # 1-5 frames of random actions
            # random among the available actions the teacher marks safe (survivable deviations:
            # a uniformly random JUMP or DUCK near the ceiling or an obstacle is a certain crash,
            # which cut collection episodes to ~160 frames even on L1)
            ok = np.flatnonzero((obs["action_mask"] != 0) & p.safe)
            a = int(rng.choice(ok)) if len(ok) else p.action
            burst_left -= 1
        executed[i] = a
        obs, r, term, trunc, info = env.step(a)
        out["reward"][i] = r
        i += 1
        if term or trunc:
            out["done"][i - 1] = True
            break
    for k in out:
        out[k] = out[k][:i]
    out["episode_id"][:] = int(task["episode_id"])
    out["seed"][:] = seed
    out["level"][:] = lev
    out["executed"] = executed[:i]
    if logits is not None:
        out["student_logits"] = logits[:i]
    out["outcome"] = {"level": level, "seed": seed, "episode_id": int(task["episode_id"]),
                      "frames": i, "distance_m": float(info["distance_m"]),
                      "crash_cause": info["crash_cause"]}
    return out
