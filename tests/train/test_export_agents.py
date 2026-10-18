"""ONNX export parity, the onnx-* agents in the app registry, and MLP action masking."""
import json

import numpy as np
import pytest
import torch

from app.agents_base import REGISTRY, Agent, Decision, load_agent
from drone_env.env import DroneDashEnv
from models import build_model, save_checkpoint
from train.export_onnx import ARC_INPUTS, ARC_OUTPUTS, export


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("export")
    torch.manual_seed(0)
    # arc.onnx: a v2.3 pilot (13-value hazard rows, ttc features, wider FFN); old.onnx base: an 8-value one
    arc = build_model("arc", {"d": 32, "heads": 4, "enc_layers": 1, "head_layers": 1, "haz_cols": 13,
                              "ttc_features": True, "ffn_mult": 4})
    legacy, mlp = build_model("arc", {"d": 32, "heads": 4, "enc_layers": 1, "head_layers": 1}), build_model("mlp")
    with torch.no_grad():                         # make the MLP love JUMP so masking is visible
        mlp.pi.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 50.0, 0.0]))
    save_checkpoint(d / "arc.pt", arc, "arc", stage="supervised", config={"data": []})
    save_checkpoint(d / "legacy.pt", legacy, "arc", stage="supervised", config={"data": []})
    save_checkpoint(d / "mlp.pt", mlp, "mlp", stage="supervised", config={"data": []})
    cal = d / "cal.json"
    cal.write_text(json.dumps({"temperature": 1.5, "tau": 0.4}))
    s_arc = export(str(d / "arc.pt"), str(d / "arc.onnx"), "arc", str(cal), parity_n=200)
    s_leg = export(str(d / "legacy.pt"), str(d / "legacy.onnx"), "arc", str(cal), parity_n=200)
    s_mlp = export(str(d / "mlp.pt"), str(d / "mlp.onnx"), "mlp", None, parity_n=200)
    return d, s_arc, s_mlp, s_leg


def test_onnx_io_names_and_parity(exported):
    import onnx
    d, s_arc, s_mlp, s_leg = exported
    m = onnx.load(str(d / "arc.onnx"))
    assert [i.name for i in m.graph.input] == ARC_INPUTS
    assert [o.name for o in m.graph.output] == ARC_OUTPUTS
    assert m.opset_import[0].version == 18
    types = {i.name: i.type.tensor_type.elem_type for i in m.graph.input}
    assert types["hazard_type"] == onnx.TensorProto.INT64 and types["act_ids"] == onnx.TensorProto.INT64
    assert types["hazard_mask"] == onnx.TensorProto.BOOL and types["act_avail"] == onnx.TensorProto.BOOL
    shapes = {o.name: [x.dim_value for x in o.type.tensor_type.shape.dim] for o in m.graph.output}
    assert shapes == {"logits": [1, 5], "safe_logits": [1, 5], "threat_logits": [1, 5], "decide_attn": [1, 12]}
    mm = onnx.load(str(d / "mlp.onnx"))
    assert [i.name for i in mm.graph.input] == ["obs"] and [o.name for o in mm.graph.output] == ["logits"]
    hz = {i.name: [x.dim_value for x in i.type.tensor_type.shape.dim] for i in m.graph.input}["hazards"]
    assert hz == [1, 12, 13]
    for s in (s_arc, s_mlp, s_leg):
        assert s["parity"]["pass"] and s["parity"]["max_abs_diff_logits"] <= 1e-4
    side = json.loads((d / "arc.json").read_text())
    from drone_env.constants import SPEC_VERSION
    assert side["temperature"] == 1.5 and side["tau"] == 0.4 and side["spec_version"] == SPEC_VERSION == "2.3"
    assert json.loads((d / "legacy.json").read_text())["spec_version"] == "2.2"     # 8-value rows
    assert side["trained_from"].endswith("arc.pt") and "val_ratio" in side
    assert json.loads((d / "mlp.json").read_text())["temperature"] == 1.0


def test_registry_loads_onnx_agents_and_fills_the_decision(exported):
    d, *_ = exported
    a = load_agent(f"onnx-arc:{d / 'arc.onnx'}")
    assert "onnx-arc" in REGISTRY and "onnx-mlp" in REGISTRY and isinstance(a, Agent)
    env = DroneDashEnv("L3")
    obs, _ = env.reset(seed=100_000)
    a.reset(env.action_descriptors())
    # v1.7: the untrained net DUCKs (now a dive) into the ground before any hazard is in view, so
    # the rule agent flies and the ONNX agent decides on every observation it sees.
    pilot = load_agent("rule")
    pilot.reset(env.action_descriptors())
    seen_attended = False
    for _ in range(240):
        dec = a.act(obs)
        assert isinstance(dec, Decision) and obs["action_mask"][dec.action] == 1
        assert dec.probs.shape == (5,) and dec.probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(dec.probs[obs["action_mask"] == 0] == 0)
        assert dec.p_safe.shape == (5,) and 0 <= dec.threat <= 4
        assert dec.hesitating == bool(dec.probs.max() < 0.4)
        if dec.attended is not None:
            assert obs["hazard_mask"][dec.attended] == 1
            seen_attended = True
        obs, _, term, _, _ = env.step(pilot.act(obs).action)
        if term:
            break
    assert seen_attended
    with pytest.raises(ValueError):
        load_agent("onnx-arc:does/not/exist.onnx")


def test_mlp_agent_masks_unavailable_actions(exported):
    d, *_ = exported
    a = load_agent(f"onnx-mlp:{d / 'mlp.onnx'}")
    env = DroneDashEnv("L1")
    obs, _ = env.reset(seed=100_000)
    a.reset(env.action_descriptors())
    assert a.act(obs).action == 3                       # JUMP available: the biased MLP takes it
    obs, *_ = env.step(3)                               # now in the jump burst: only HOLD
    dec = a.act(obs)
    assert dec.action == 0 and dec.probs[3] == 0 and dec.probs[0] == pytest.approx(1.0)
    for _ in range(12):
        obs, *_ = env.step(0)
    assert obs["action_mask"][3] == 0                   # cooldown: JUMP masked, others allowed
    dec = a.act(obs)
    assert dec.action != 3 and dec.probs[3] == 0


def test_pre_v2_pilots_see_new_obstacles_as_kinds_they_know(exported):
    """A sidecar from before SPEC v2.0 means an 8-row type embedding: v2.0 kinds are mapped onto
    building (ground obstacles) and plane (bands in the air); current pilots get the ids unchanged."""
    from agents.onnx_agents import pre_v2_type_map
    from drone_env import constants as C
    m = pre_v2_type_map()
    assert m.max() < 8 and list(m[:8]) == list(range(8))
    assert m[C.TREE] == m[C.POLE] == m[C.KITE] == C.BUILDING and m[C.POWERLINE] == m[C.CRANE] == C.PLANE
    d, *_ = exported
    a = load_agent(f"onnx-arc:{d / 'arc.onnx'}")
    assert a.type_map is None
    side = json.loads((d / "legacy.json").read_text())
    (d / "old.json").write_text(json.dumps({**side, "spec_version": "1.9"}))
    (d / "old.onnx").write_bytes((d / "legacy.onnx").read_bytes())
    old = load_agent(f"onnx-arc:{d / 'old.onnx'}")
    env = DroneDashEnv("L1")
    env.sensor_spec = old.sensor_spec
    obs, _ = env.reset(seed=100_000)
    old.reset(env.action_descriptors())
    obs["hazard_type"][:5] = [C.TREE, C.POLE, C.POWERLINE, C.KITE, C.CRANE]
    obs["hazard_mask"][:5] = 1
    old.act(obs)
    assert list(old.feed["hazard_type"][0, :5]) == [C.BUILDING, C.BUILDING, C.PLANE, C.BUILDING, C.PLANE]


def test_each_pilot_gets_the_sensor_spec_it_was_made_for(exported):
    """v2.3 pilots read tracker rows (13 values, 20 m); a v2.2 sidecar means band rows, older ones
    body rows (8 values, 16 m). The app and eval.evaluate set env.sensor_spec from the agent."""
    from drone_env import constants as C
    from eval.evaluate import fly_agent
    d, *_ = exported
    a = load_agent(f"onnx-arc:{d / 'arc.onnx'}")
    assert a.sensor_spec == (2, 3) and a.feed["hazards"].shape == (1, 12, 13)
    side = json.loads((d / "legacy.json").read_text())
    for v, want in (("2.1", (2, 1)), ("1.9", (1, 9)), ("2.2", (2, 2))):
        (d / "v.json").write_text(json.dumps({**side, "spec_version": v}))
        (d / "v.onnx").write_bytes((d / "legacy.onnx").read_bytes())
        ag = load_agent(f"onnx-arc:{d / 'v.onnx'}")
        assert ag.sensor_spec == want and ag.feed["hazards"].shape == (1, 12, C.hazard_cols(want)), v
    assert load_agent(f"onnx-mlp:{d / 'mlp.onnx'}").sensor_spec == (2, 2)
    for name in ("arc", "legacy"):                # both fly in eval without shape errors
        r = fly_agent((f"onnx-arc:{d / (name + '.onnx')}", "L1", 100_000, 60))
        assert r["frames"] > 0
