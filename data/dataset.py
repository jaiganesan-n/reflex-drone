"""Shard format (SPEC §12) and the training-time loader.

* ``data/<name>/shard_XXXX.npz`` + ``data/<name>/meta.json``. A shard holds whole episodes and
  closes at the first episode boundary at or after ``shard_frames`` (50,000) frames.
* Train / holdout split by episode (5% holdout, chosen by a hash of the episode's seed, so the
  split is stable when shards are added or datasets are merged).
* Weighted sampling: threat >= 2 counts x3; DAgger frames whose student confidence is below
  tau count x2.
* MLP view: ``drone_env.wrappers.flatten_obs`` per frame plus the ``[prev, cur]`` stack rebuilt
  from ``episode_id`` and ``step`` (first frame duplicated).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from drone_env import constants as C

FIELDS = {
    "self": np.float32, "rays": np.float16, "hazards": np.float16, "hazard_type": np.int8,
    "hazard_mask": np.bool_, "action_mask": np.bool_, "teacher_scores": np.float32, "safe": np.bool_,
    "threat": np.int8, "action": np.int8, "student_action": np.int8, "student_conf": np.float16,
    "reward": np.float32, "done": np.bool_, "episode_id": np.int32, "step": np.int32, "seed": np.int32,
    "level": np.int8, "tags": np.bool_,
}
SHAPES = {"self": (11,), "rays": (13,), "hazards": (12, C.HAZ_COLS), "hazard_type": (12,), "hazard_mask": (12,),
          "action_mask": (5,), "teacher_scores": (5,), "safe": (5,), "tags": (3,)}
SHARD_FRAMES = 50_000
HOLDOUT_EVERY = 20                     # 1 in 20 episodes = 5%


# ------------------------------------------------------------------------------------------
# files
# ------------------------------------------------------------------------------------------
def shard_path(d: str | Path, k: int) -> Path:
    return Path(d) / f"shard_{k:04d}.npz"


def list_shards(d: str | Path) -> list[Path]:
    return sorted(Path(d).glob("shard_*.npz"))


def write_shard(path: str | Path, arrays: dict) -> None:
    """Write one shard with the exact SPEC dtypes (atomic: tmp file, then rename)."""
    path = Path(path)
    out = {}
    for k, dt in FIELDS.items():
        a = np.asarray(arrays[k]).astype(dt, copy=False)
        legacy = k == "hazards" and a.shape[1:] == (12, C.LEGACY_HAZ_COLS)     # data recorded before v2.3
        if k in SHAPES and a.shape[1:] != SHAPES[k] and not legacy:
            raise ValueError(f"{k}: shape {a.shape} does not match SPEC {SHAPES[k]}")
        out[k] = a
    n = {len(v) for v in out.values()}
    if len(n) != 1:
        raise ValueError(f"ragged shard arrays: {n}")
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(path)


def read_shard(path: str | Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def concat(parts: list[dict], keys=None) -> dict:
    keys = keys or list(FIELDS)
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def read_meta(d: str | Path) -> dict | None:
    p = Path(d) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_meta(d: str | Path, meta: dict) -> None:
    p = Path(d) / "meta.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(p)


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def load_frames(dirs) -> dict:
    """Concatenate every shard of one or more dataset dirs. Adds ``ep_key`` (int64, unique per
    episode across dirs) and ``source`` (index of the dir)."""
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]
    parts = []
    for si, d in enumerate(dirs):
        shards = list_shards(d)
        if not shards:
            raise FileNotFoundError(f"no shards in {d}")
        for s in shards:
            a = read_shard(s)
            a["source"] = np.full(len(a["step"]), si, np.int16)
            parts.append(a)
    out = concat(parts, list(FIELDS) + ["source"])
    out["ep_key"] = out["source"].astype(np.int64) * (1 << 32) + out["episode_id"].astype(np.int64)
    return out


# ------------------------------------------------------------------------------------------
# split, weights, MLP view (numpy)
# ------------------------------------------------------------------------------------------
def holdout_mask(seed: np.ndarray) -> np.ndarray:
    """True for holdout frames: 1 episode in 20, by a hash of its seed."""
    s = np.asarray(seed, dtype=np.uint64)
    h = (s * np.uint64(2654435761) + np.uint64(97)) % np.uint64(1 << 32)
    return (h >> np.uint64(7)) % np.uint64(HOLDOUT_EVERY) == 0


def sample_weights(arr: dict, tau: float | None = None, burst_weight: float = 1.0) -> np.ndarray:
    """threat >= 2 -> x3; DAgger frames (student_action >= 0) with student_conf < tau -> x2;
    teacher action JUMP or DUCK -> x burst_weight (session 3: the bursts are ~0.5% of labels, and a
    student trained without this almost never uses them)."""
    w = np.ones(len(arr["threat"]), np.float64)
    w[arr["threat"] >= 2] *= 3.0
    if tau is not None and tau > 0:
        hes = (arr["student_action"] >= 0) & (arr["student_conf"].astype(np.float32) < tau)
        w[hes] *= 2.0
    if burst_weight != 1.0:
        w[np.isin(arr["action"], (3, 4))] *= float(burst_weight)     # JUMP, DUCK
    return w


def flat_view(arr: dict, chunk: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """(flat (N, 40) float32 via drone_env.wrappers.flatten_obs, prev_idx (N,) int64).

    The MLP input for frame i is ``concatenate([flat[prev_idx[i]], flat[i]])``: the previous
    frame of the same episode (step - 1), or the frame itself at an episode's first frame."""
    from drone_env.wrappers import flatten_obs
    n = len(arr["step"])
    flat = np.empty((n, 40), np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        flat[a:b] = flatten_obs(arr["self"][a:b], arr["rays"][a:b], arr["hazards"][a:b],
                                arr["hazard_type"][a:b], arr["hazard_mask"][a:b])
    key = arr["ep_key"] if "ep_key" in arr else arr["episode_id"].astype(np.int64)
    step = arr["step"].astype(np.int64)
    order = np.lexsort((step, key))
    prev = np.arange(n, dtype=np.int64)
    ks, ss = key[order], step[order]
    cont = np.zeros(n, bool)
    cont[1:] = (ks[1:] == ks[:-1]) & (ss[1:] == ss[:-1] + 1)
    prev[order[1:][cont[1:]]] = order[:-1][cont[1:]]
    return flat, prev


def label_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1.0)
    return -(np.where(p > 0, p * np.log(q), 0.0)).sum(-1)


# ------------------------------------------------------------------------------------------
# torch view for training
# ------------------------------------------------------------------------------------------
class FrameData:
    """All frames as CPU torch tensors, with train / holdout indices and sampling weights.

    ``batch(idx)`` returns a dict of tensors for Arc (``mlp=False``) or the MLP
    (``mlp=True``: ``obs`` (B, 80)), plus the label tensors.
    """

    def __init__(self, dirs, tau: float | None = None, mlp: bool = False, label_temperature: float = 2.0,
                 burst_weight: float = 1.0):
        import torch
        from teacher.planner import soft_targets
        arr = load_frames(dirs)
        self.n = len(arr["step"])
        self.dirs = [str(d) for d in ([dirs] if isinstance(dirs, (str, Path)) else dirs)]
        hold = holdout_mask(arr["seed"])
        if not hold.any():
            # tiny datasets (smoke): hold out the episode with the largest seed, if there are two
            if len(np.unique(arr["ep_key"])) >= 2:
                hold = arr["ep_key"] == arr["ep_key"][np.argmax(arr["seed"])]
        if hold.all():
            hold = np.zeros_like(hold)
        self.holdout_is_train = not hold.any()
        if self.holdout_is_train:
            hold_idx = np.arange(len(hold))
        else:
            hold_idx = np.flatnonzero(hold)
        self.train_idx = torch.from_numpy(np.flatnonzero(~hold))
        self.holdout_idx = torch.from_numpy(hold_idx)
        self.weights = torch.from_numpy(sample_weights(arr, tau, burst_weight))
        t = lambda a, dt=None: torch.from_numpy(np.ascontiguousarray(a if dt is None else a.astype(dt)))
        self.t = {
            "self": t(arr["self"], np.float32),
            "rays": t(arr["rays"], np.float32),
            "hazards": t(arr["hazards"], np.float32),
            "hazard_type": t(arr["hazard_type"], np.int64),
            "hazard_mask": t(arr["hazard_mask"], np.bool_),
            "act_avail": t(arr["action_mask"], np.bool_),
            "p_teacher": t(soft_targets(arr["teacher_scores"], label_temperature)),
            "safe": t(arr["safe"], np.float32),
            "threat": t(arr["threat"], np.int64),
            "action": t(arr["action"], np.int64),
        }
        self.level = arr["level"]
        self.mlp = mlp
        if mlp:
            flat, prev = flat_view(arr)
            self.t["flat"] = t(flat)
            self.t["prev"] = t(prev)
        self.meta = [read_meta(d) for d in self.dirs]

    def batch(self, idx, device=None) -> dict:
        import torch
        keys = ("act_avail", "p_teacher", "safe", "threat", "action")
        b = {k: self.t[k][idx] for k in keys}
        if self.mlp:
            b["obs"] = torch.cat([self.t["flat"][self.t["prev"][idx]], self.t["flat"][idx]], dim=1)
        else:
            for k in ("self", "rays", "hazards", "hazard_type", "hazard_mask"):
                b[k] = self.t[k][idx]
        if device is not None and str(device) != "cpu":
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        return b

    def epoch_indices(self, generator=None):
        """Weighted sample (with replacement) of len(train) training frames."""
        import torch
        w = self.weights[self.train_idx]
        pick = torch.multinomial(w, len(self.train_idx), replacement=True, generator=generator)
        return self.train_idx[pick]
