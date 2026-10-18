"""Shards (SPEC §12): round trip, resume and top-up; weighted sampling; the MLP view."""
import json
import re

import numpy as np
import pytest
import torch

from data.collect import collect, load_collect_config
from data.dataset import (FIELDS, SHAPES, FrameData, flat_view, holdout_mask, list_shards, load_frames, read_meta,
                          read_shard, sample_weights)
from drone_env.env import DroneDashEnv
from drone_env.wrappers import FlatHistory


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("data") / "tiny"
    cfg = load_collect_config("configs/train/smoke_collect.yaml")
    cfg.update(frames_cap=40, shard_frames=60, workers=1)
    collect(cfg, d, target_frames=100, workers=1, quiet=True)
    return d, cfg


def test_shard_round_trip_has_exact_spec_dtypes(tiny):
    d, _ = tiny
    shards = list_shards(d)
    assert len(shards) >= 2
    for s in shards:
        a = read_shard(s)
        assert set(a) == set(FIELDS)
        n = len(a["step"])
        for k, dt in FIELDS.items():
            assert a[k].dtype == np.dtype(dt), k
            assert a[k].shape[0] == n and a[k].shape[1:] == SHAPES.get(k, ()), k
        assert (a["student_action"] == -1).all()
        assert a["done"].sum() == len(np.unique(a["episode_id"]))      # whole episodes per shard
        assert ((a["seed"] >= 100_000) & (a["seed"] < 100_050)).all()    # smoke config: VAL seeds
        assert np.all(a["action_mask"][np.arange(n), a["action"]])       # labels are available actions
    meta = read_meta(d)
    assert meta["total_frames"] == sum(s["frames"] for s in meta["shards"]) >= 100
    assert meta["beam"] >= 1 and len(meta["config_hash"]) == 64
    assert sorted(meta["seeds_used"]) == sorted({x for s in meta["shards"] for x in s["seeds"]})


def test_rerun_is_a_noop_and_topup_adds_new_seeds(tiny):
    d, cfg = tiny
    before = {p.name: p.read_bytes() for p in list_shards(d)}
    meta0 = read_meta(d)
    collect(cfg, d, target_frames=100, workers=1, quiet=True)          # already complete
    assert {p.name: p.read_bytes() for p in list_shards(d)} == before
    collect(cfg, d, target_frames=meta0["total_frames"] + 50, workers=1, quiet=True)
    meta1 = read_meta(d)
    after = {p.name: p.read_bytes() for p in list_shards(d)}
    assert all(after[k] == v for k, v in before.items())                # old shards untouched
    assert len(after) > len(before)
    new_seeds = set(meta1["seeds_used"]) - set(meta0["seeds_used"])
    assert new_seeds and min(new_seeds) > max(meta0["seeds_used"])
    # a different teacher is refused; a different burst_prob is a legal top-up
    bad = d.parent / "teacher_k1.yaml"
    bad.write_text(re.sub(r"^beam: 2\b", "beam: 1", open("configs/train/teacher.yaml").read(), flags=re.M))
    with pytest.raises(SystemExit):
        collect(dict(cfg, teacher_config=str(bad)), d, target_frames=10_000, workers=1, quiet=True)
    collect(dict(cfg, burst_prob=0.3), d, target_frames=meta1["total_frames"] + 20, workers=1, quiet=True)
    meta2 = read_meta(d)
    assert meta2["shards"][-1]["burst_prob"] == 0.3 and meta2["shards"][0]["burst_prob"] == cfg["burst_prob"]


def test_weighted_sampler():
    n = 1000
    arr = {"threat": np.zeros(n, np.int8), "student_action": np.full(n, -1, np.int8),
           "student_conf": np.ones(n, np.float16)}
    arr["threat"][:100] = 2
    arr["student_action"][900:] = 1
    arr["student_conf"][900:950] = 0.3
    w = sample_weights(arr, tau=0.5)
    assert np.all(w[:100] == 3) and np.all(w[100:900] == 1)
    assert np.all(w[900:950] == 2) and np.all(w[950:] == 1)
    assert np.all(sample_weights(arr, tau=None)[900:] == 1)
    # the multinomial draw follows the weights
    g = torch.Generator().manual_seed(0)
    idx = torch.multinomial(torch.from_numpy(w), 200_000, replacement=True, generator=g).numpy()
    share_hi = np.isin(idx, np.arange(100)).mean()
    assert share_hi == pytest.approx(300 / w.sum(), rel=0.05)


def test_burst_weight_upweights_teacher_jump_and_duck():
    n = 10
    arr = {"threat": np.zeros(n, np.int8), "student_action": np.full(n, -1, np.int8),
           "student_conf": np.ones(n, np.float16), "action": np.array([0, 1, 2, 3, 4] * 2, np.int8)}
    arr["threat"][3] = 2
    w = sample_weights(arr, burst_weight=10.0)
    assert list(w[:5]) == [1, 1, 1, 30, 10] and list(w[5:]) == [1, 1, 1, 10, 10]
    assert np.all(sample_weights(arr)[[0, 1, 2, 4]] == 1)          # default 1.0: unchanged


def test_holdout_is_about_five_percent_and_stable():
    seeds = np.arange(0, 100_000)
    m = holdout_mask(seeds)
    assert 0.045 < m.mean() < 0.055
    assert np.array_equal(m[:500], holdout_mask(np.arange(0, 500)))


def test_mlp_view_matches_flathistory(tiny):
    d, _ = tiny
    a = load_frames(d)
    flat, prev = flat_view(a)
    # rebuild one episode by flying it again and stacking with FlatHistory (float16 storage aside)
    ep = a["episode_id"] == a["episode_id"][0]
    steps = a["step"][ep]
    assert steps[0] == 0
    i0 = np.flatnonzero(ep)
    assert prev[i0[0]] == i0[0]                                          # first frame duplicated
    assert np.all(prev[i0[1:]] == i0[:-1])
    x = np.concatenate([flat[prev[i0]], flat[i0]], 1)
    assert x.shape[1] == 80
    assert np.allclose(x[0, :40], x[0, 40:])
    fd = FrameData(d, mlp=True)
    b = fd.batch(torch.from_numpy(i0[:5]))
    assert b["obs"].shape == (5, 80) and torch.allclose(b["obs"], torch.from_numpy(x[:5]))
    h = FlatHistory()
    h.reset()
    env = DroneDashEnv("L1")
    obs, _ = env.reset(seed=1)
    v0 = h.push(obs)
    assert np.allclose(v0[:40], v0[40:])
