"""Report refuses mixed groups; planner caches merge; calibration maths."""
import json

import numpy as np
import pytest

from eval import planner_cache as pc
from eval.report import MixedGroups, build, load_results
from train.calibrate import _softmax, ece, fit_temperature, nll, pick_tau


def _res(path, group, agent="onnx-arc:artifacts/x.onnx", level="L5", seeds=(1, 2)):
    d = {"group": group, "split": "val" if group == "val" else "test", "levels": [level], "agent": agent,
         "agent_info": {"arch": "arc", "stage": "dagger"}, "frame_cap": 6000,
         "summary": {level: {"mean_distance_m": 100.0, "hesitation_share": 0.1}},
         "per_seed": [{"level": level, "seed": s, "distance_m": 100.0 + s, "frames": 10, "crash_cause": "bird",
                       "hesitation_share": 0.1} for s in seeds]}
    path.write_text(json.dumps(d))
    return path


def test_report_refuses_mixed_groups(tmp_path):
    a = _res(tmp_path / "a.json", "val")
    b = _res(tmp_path / "b.json", "final", agent="planner")
    with pytest.raises(MixedGroups):
        load_results([a, b])
    from eval.report import main
    assert main([str(a), str(b), "--out", str(tmp_path / "r.md")]) == 2
    assert not (tmp_path / "r.md").exists()


def test_report_builds_ratios_for_one_group(tmp_path):
    p = _res(tmp_path / "p.json", "final", agent="planner")
    m = _res(tmp_path / "m.json", "final")
    text = build([p, m], str(tmp_path / "results.md"))
    assert "FINAL group" in text and "Arc + DAgger" in text and "| 1.000 |" in text


def test_planner_cache_merges_and_never_drops_entries(tmp_path):
    path = pc.cache_path(tmp_path, "val")
    pc.merge(path, {"L1:100000": {"distance_m": 1.0}}, "val")
    pc.merge(path, {"L2:100001": {"distance_m": 2.0}}, "val")
    pc.merge(path, {"L1:100000": {"distance_m": 3.0}}, "val")
    e = pc.load_cache(path)["entries"]
    assert e == {"L1:100000": {"distance_m": 3.0}, "L2:100001": {"distance_m": 2.0}}
    with pytest.raises(ValueError):
        pc.ensure(tmp_path, "val", "L1", [200_000], 600, 1)        # a TEST seed is not in the val group


def test_temperature_fit_recovers_a_known_temperature():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, size=(4000, 5))
    p_true = _softmax(z, 2.0)
    y = np.array([rng.choice(5, p=p) for p in p_true])
    best = np.zeros_like(z, bool)
    best[np.arange(len(y)), y] = True
    T = fit_temperature(z, best)
    assert 1.7 < T < 2.3
    assert nll(z, best, T) <= nll(z, best, 1.0)
    conf = _softmax(z, T).max(-1)
    correct = best[np.arange(len(y)), z.argmax(-1)]
    assert ece(conf, correct) < ece(_softmax(z, 1.0).max(-1), correct)


def test_pick_tau():
    conf = np.linspace(0.2, 1.0, 1000)
    correct = conf > 0.5                                     # below 0.5 always wrong
    tau = pick_tau(conf, correct)
    assert 0.5 < tau < 0.51 and correct[conf < tau].mean() < 0.7 and correct[conf >= tau].mean() >= 0.7
    assert pick_tau(conf, np.ones(1000, bool)) == 0.0         # always agrees: never hesitant
    assert pick_tau(conf, np.zeros(1000, bool)) == 1.0        # never agrees: always hesitant
    rng = np.random.default_rng(1)
    noisy = rng.random(5000) < np.clip(conf.repeat(5) * 1.2 - 0.2, 0, 1)   # agreement = 1.2 c - 0.2
    t2 = pick_tau(conf.repeat(5), noisy)
    assert 0.65 < t2 < 0.85                                  # true crossing at c = 0.75
