"""ONNX Runtime agents for the app and the evaluation (SPEC §8, §13).

* ``onnx-arc:<path.onnx>``: Arc. Fills P(safe), threat, the attended hazard
  and the hesitation flag for the HUD.
* ``onnx-mlp:<path.onnx>``: the flat MLP baseline with its own ``[prev, cur]`` buffer.

Both read the sidecar with the same stem (``arc.onnx`` -> ``arc.json``) for the temperature T
and the hesitation threshold tau, run ONNX Runtime on one thread, and see only the
observation dict. One forward pass per decision; no search, no planner.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.agents_base import Decision
from drone_env import constants as C
from drone_env.wrappers import FlatHistory

ARC_INPUTS = ("self", "rays", "hazards", "hazard_type", "hazard_mask", "act_ids", "act_desc", "act_avail")
ARC_OUTPUTS = ("logits", "safe_logits", "threat_logits", "decide_attn")


def pre_v2_type_map() -> np.ndarray:
    """Hazard type ids as a pilot trained before SPEC v2.0 knows them (8 kinds): trees, street lights
    and kites read as buildings (blocked from the ground up), power lines and cranes as planes (a band
    in the air). Lets older pilots run in the v2.0 game; they were never trained on these obstacles."""
    m = np.arange(C.N_HAZARD_TYPES, dtype=np.int64)
    m[[C.TREE, C.POLE, C.KITE]] = C.BUILDING
    m[[C.POWERLINE, C.CRANE]] = C.PLANE
    return m


def _version(sidecar: dict) -> tuple:
    try:
        return tuple(int(x) for x in str(sidecar.get("spec_version", C.SPEC_VERSION)).split(".")[:2])
    except ValueError:
        return tuple(int(x) for x in C.SPEC_VERSION.split(".")[:2])


def sensor_spec_for(sidecar: dict) -> tuple:
    """Sensor spec a pilot was made for (its sidecar's spec_version, capped at the current spec):
    v2.3 tracker rows, v2.2 band rows, or the body rows before that (drone_env.sensors)."""
    v = _version(sidecar)
    return v if v < C.SENSOR_SPEC else C.SENSOR_SPEC


def _pre_v2(sidecar: dict) -> bool:
    try:
        return int(str(sidecar.get("spec_version", C.SPEC_VERSION)).split(".")[0]) < 2
    except ValueError:
        return False


def read_sidecar(onnx_path) -> dict:
    p = Path(onnx_path).with_suffix(".json")
    if not p.exists():
        return {"temperature": 1.0, "tau": 0.0}
    return json.loads(p.read_text())


def make_session(onnx_path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])


def _softmax_masked(z: np.ndarray, avail: np.ndarray, T: float) -> np.ndarray:
    s = np.where(avail, z / T, -np.inf)
    m = s.max()
    e = np.where(avail, np.exp(s - m), 0.0)
    return (e / e.sum()).astype(np.float32)


class OnnxArcAgent:
    def __init__(self, path: str):
        if not path:
            raise ValueError("onnx-arc needs a model path: onnx-arc:artifacts/arc.onnx")
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(f"onnx-arc: {self.path} not found")
        self.name = f"onnx-arc:{path}"
        self.sidecar = read_sidecar(self.path)
        self.T = float(self.sidecar.get("temperature", 1.0) or 1.0)
        self.tau = float(self.sidecar.get("tau", 0.0) or 0.0)
        self.sess = make_session(self.path)
        names = [i.name for i in self.sess.get_inputs()]
        if tuple(sorted(names)) != tuple(sorted(ARC_INPUTS)):
            raise ValueError(f"{self.path}: inputs {names} are not the SPEC §13 Arc inputs")
        self.out_names = list(ARC_OUTPUTS)
        self.type_map = pre_v2_type_map() if _pre_v2(self.sidecar) else None
        self.sensor_spec = sensor_spec_for(self.sidecar)     # the hazard rows it was trained on
        self.feed = {
            "self": np.zeros((1, 11), np.float32), "rays": np.zeros((1, 13), np.float32),
            "hazards": np.zeros((1, 12, C.hazard_cols(self.sensor_spec)), np.float32),
            "hazard_type": np.zeros((1, 12), np.int64),
            "hazard_mask": np.zeros((1, 12), bool), "act_ids": np.arange(5, dtype=np.int64)[None],
            "act_desc": np.zeros((1, 5, 3), np.float32), "act_avail": np.ones((1, 5), bool),
        }

    def reset(self, action_desc: np.ndarray) -> None:
        self.feed["act_desc"][0] = np.asarray(action_desc, np.float32)

    def fill(self, obs: dict) -> None:
        f = self.feed
        f["self"][0] = obs["self"]
        f["rays"][0] = obs["rays"]
        f["hazards"][0] = obs["hazards"]
        f["hazard_type"][0] = obs["hazard_type"] if self.type_map is None else self.type_map[obs["hazard_type"]]
        np.not_equal(obs["hazard_mask"], 0, out=f["hazard_mask"][0])
        np.not_equal(obs["action_mask"], 0, out=f["act_avail"][0])

    def run(self):
        return self.sess.run(self.out_names, self.feed)

    def act(self, obs: dict) -> Decision:
        self.fill(obs)
        logits, safe, threat, attn = self.run()
        avail = self.feed["act_avail"][0]
        probs = _softmax_masked(logits[0], avail, self.T)
        a = int(np.argmax(probs))
        hm = self.feed["hazard_mask"][0]
        attended = int(np.argmax(np.where(hm, attn[0], -1.0))) if hm.any() else None
        return Decision(action=a, probs=probs, p_safe=(1.0 / (1.0 + np.exp(-safe[0]))).astype(np.float32),
                        threat=int(np.argmax(threat[0])), attended=attended,
                        hesitating=bool(probs[a] < self.tau))


class OnnxMLPAgent:
    def __init__(self, path: str):
        if not path:
            raise ValueError("onnx-mlp needs a model path: onnx-mlp:artifacts/mlp.onnx")
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(f"onnx-mlp: {self.path} not found")
        self.name = f"onnx-mlp:{path}"
        self.sidecar = read_sidecar(self.path)
        self.T = float(self.sidecar.get("temperature", 1.0) or 1.0)
        self.tau = float(self.sidecar.get("tau", 0.0) or 0.0)
        self.sess = make_session(self.path)
        names = [i.name for i in self.sess.get_inputs()]
        if names != ["obs"]:
            raise ValueError(f"{self.path}: inputs {names} are not the SPEC §13 MLP input ['obs']")
        self.out_names = ["logits"]
        self.hist = FlatHistory()
        self.sensor_spec = sensor_spec_for(self.sidecar)
        self.feed = {"obs": np.zeros((1, 80), np.float32)}
        self._avail = np.ones(5, bool)

    def reset(self, action_desc: np.ndarray) -> None:
        self.hist.reset()

    def fill(self, obs: dict) -> None:
        self.feed["obs"][0] = self.hist.push(obs)
        np.not_equal(obs["action_mask"], 0, out=self._avail)

    def run(self):
        return self.sess.run(self.out_names, self.feed)

    def act(self, obs: dict) -> Decision:
        self.fill(obs)
        (logits,) = self.run()
        probs = _softmax_masked(logits[0], self._avail, self.T)     # masked logits = -inf
        a = int(np.argmax(probs))
        return Decision(action=a, probs=probs, hesitating=bool(probs[a] < self.tau))
