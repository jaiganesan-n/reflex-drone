"""Export a checkpoint to ONNX with the SPEC §13 I/O names, plus its sidecar.

    uv run python -m train.export_onnx [--arch mlp] --model runs/<run>/best.pt \
        [--calibration runs/<run>/calibration.json] --out artifacts/<stem>.onnx

* Arc inputs (batch 1): self f32[1,11], rays f32[1,13], hazards f32[1,12,8], hazard_type
  i64[1,12], hazard_mask bool[1,12], act_ids i64[1,5], act_desc f32[1,5,3], act_avail bool[1,5].
  Outputs: logits f32[1,5], safe_logits f32[1,5], threat_logits f32[1,5], decide_attn f32[1,12].
  (The prototype's names differ; they are renamed here. The value head is not exported.)
* MLP (``--arch mlp``): input obs f32[1,80], output logits f32[1,5] (unmasked).
* Opset 18 through the dynamo exporter (needs onnxscript).
* Sidecar ``artifacts/<stem>.json``: temperature, tau (1.0 / 0 without --calibration),
  spec_version, val_ratio (from the run's summary.json), trained_from (the checkpoint path).
* Parity check: ONNX Runtime vs PyTorch logits within 1e-4 on 1,000 holdout frames of the
  run's training data (or, if that data is gone, 1,000 frames flown on VAL seeds).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from drone_env import constants as C
from models import load_checkpoint

ARC_INPUTS = ["self", "rays", "hazards", "hazard_type", "hazard_mask", "act_ids", "act_desc", "act_avail"]
ARC_OUTPUTS = ["logits", "safe_logits", "threat_logits", "decide_attn"]
PARITY_ATOL = 1e-4


class ArcExport(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, self_f, rays, hazards, hazard_type, hazard_mask, act_ids, act_desc, act_avail):
        lg, sf, th, attn, _v = self.net(self_f, rays, hazards, hazard_type, hazard_mask, act_ids, act_desc, act_avail)
        return lg, sf, th, attn


class MLPExport(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        return self.net(obs)[0]


def _dummy_arc(haz_cols: int = 8):
    from drone_env.env import action_descriptors
    hm = torch.zeros(1, 12, dtype=torch.bool)
    hm[0, :4] = True
    return (torch.zeros(1, 11), torch.full((1, 13), 0.5), torch.zeros(1, 12, haz_cols), torch.ones(1, 12, dtype=torch.int64),
            hm, torch.arange(5, dtype=torch.int64)[None], torch.from_numpy(action_descriptors())[None],
            torch.ones(1, 5, dtype=torch.bool))


def _export(module, args, out: Path, in_names, out_names) -> None:
    import onnx
    out.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    torch.onnx.export(module, args, str(out), input_names=in_names, output_names=out_names, opset_version=18,
                      dynamo=True, external_data=False)
    m = onnx.load(str(out))
    # make sure the graph carries exactly the SPEC names (rename if an exporter version did not)
    gi = [i.name for i in m.graph.input]
    go = [o.name for o in m.graph.output]
    ren = {}
    if gi != in_names:
        ren.update({a: b for a, b in zip(gi, in_names)})
    if go != out_names:
        ren.update({a: b for a, b in zip(go, out_names)})
    if ren:
        for node in m.graph.node:
            node.input[:] = [ren.get(x, x) for x in node.input]
            node.output[:] = [ren.get(x, x) for x in node.output]
        for v in list(m.graph.input) + list(m.graph.output):
            v.name = ren.get(v.name, v.name)
        onnx.save(m, str(out))
    onnx.checker.check_model(onnx.load(str(out)))


def parity_frames(ck: dict, n: int = 1000) -> dict:
    """Up to n holdout frames from the run's training data, else frames flown on VAL seeds."""
    from data.dataset import holdout_mask, list_shards, load_frames
    cfg = ck.get("config") or {}
    cand = list(cfg.get("data") or []) + ([cfg["replay_data"]] if isinstance(cfg.get("replay_data"), str) else
                                          list(cfg.get("replay_data") or []))
    dirs = [d for d in cand if Path(d).is_dir() and list_shards(d)]
    if dirs:
        a = load_frames(dirs)
        idx = np.flatnonzero(holdout_mask(a["seed"]))
        if len(idx) < n:
            idx = np.concatenate([idx, np.flatnonzero(~holdout_mask(a["seed"]))])
        idx = idx[:n]
        src = f"holdout frames of {dirs}"
        return {"src": src, "self": a["self"][idx], "rays": a["rays"][idx], "hazards": a["hazards"][idx],
                "hazard_type": a["hazard_type"][idx], "hazard_mask": a["hazard_mask"][idx],
                "action_mask": a["action_mask"][idx], "ep_key": a["ep_key"][idx], "step": a["step"][idx]}
    # fallback: fly the rule agent on VAL seeds, with the hazard rows the net reads
    from app.agents_base import load_agent
    from drone_env.env import DroneDashEnv
    rows = {k: [] for k in ("self", "rays", "hazards", "hazard_type", "hazard_mask", "action_mask", "ep_key", "step")}
    agent = load_agent("rule")
    haz_cols = int((ck.get("model_cfg") or {}).get("haz_cols", C.LEGACY_HAZ_COLS)) if ck.get("arch") == "arc" else C.LEGACY_HAZ_COLS
    seed = 100_000
    while len(rows["self"]) < n:
        env = DroneDashEnv("L5", max_frames=n)
        env.sensor_spec = C.SENSOR_SPEC if haz_cols == C.HAZ_COLS else (2, 2)
        obs, _ = env.reset(seed=seed)
        agent.reset(env.action_descriptors())
        t = 0
        while len(rows["self"]) < n:
            for k in ("self", "rays", "hazards", "hazard_type", "hazard_mask", "action_mask"):
                rows[k].append(obs[k])
            rows["ep_key"].append(seed)
            rows["step"].append(t)
            obs, _, term, trunc, _ = env.step(agent.act(obs).action)
            t += 1
            if term or trunc:
                break
        seed += 1
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["src"] = "frames flown by the rule agent on L5 VAL seeds"
    return out


def check_parity(onnx_path: Path, model, arch: str, frames: dict) -> dict:
    import onnxruntime as ort
    from drone_env.env import action_descriptors
    from drone_env.wrappers import flatten_obs
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    n = len(frames["self"])
    worst = 0.0
    worst_aux = 0.0
    mism = 0
    desc = action_descriptors()[None]
    if arch == "mlp":
        flat = flatten_obs(frames["self"], frames["rays"], frames["hazards"], frames["hazard_type"], frames["hazard_mask"])
        prev = np.arange(n)
        same = np.zeros(n, bool)
        same[1:] = (frames["ep_key"][1:] == frames["ep_key"][:-1]) & (frames["step"][1:] == frames["step"][:-1] + 1)
        prev[1:][same[1:]] = np.arange(n - 1)[same[1:]]
        X = np.concatenate([flat[prev], flat], 1).astype(np.float32)
    with torch.no_grad():
        for i in range(n):
            if arch == "mlp":
                x = X[i:i + 1]
                ref = model(torch.from_numpy(x))[0].numpy()
                got = sess.run(["logits"], {"obs": x})[0]
                outs_ref, outs_got = [ref], [got]
            else:
                feed = {"self": frames["self"][i:i + 1].astype(np.float32), "rays": frames["rays"][i:i + 1].astype(np.float32),
                        "hazards": frames["hazards"][i:i + 1].astype(np.float32),
                        "hazard_type": frames["hazard_type"][i:i + 1].astype(np.int64),
                        "hazard_mask": frames["hazard_mask"][i:i + 1].astype(bool),
                        "act_ids": np.arange(5, dtype=np.int64)[None], "act_desc": desc.astype(np.float32),
                        "act_avail": frames["action_mask"][i:i + 1].astype(bool)}
                ref = model(*(torch.from_numpy(feed[k]) for k in ARC_INPUTS))
                outs_ref = [r.numpy() for r in ref[:4]]
                outs_got = sess.run(ARC_OUTPUTS, feed)
            r, g = outs_ref[0], outs_got[0]
            if not np.array_equal(np.isfinite(r), np.isfinite(g)):
                mism += 1
                continue
            fin = np.isfinite(r)
            worst = max(worst, float(np.abs(r[fin] - g[fin]).max()) if fin.any() else 0.0)
            for a, b in zip(outs_ref[1:], outs_got[1:]):
                worst_aux = max(worst_aux, float(np.abs(a - b).max()))
    return {"frames": n, "max_abs_diff_logits": worst, "max_abs_diff_other_outputs": worst_aux,
            "mask_mismatches": mism, "atol": PARITY_ATOL, "pass": mism == 0 and worst <= PARITY_ATOL,
            "source": frames["src"]}


def export(model_path: str, out: str, arch: str = "arc", calibration: str | None = None, parity_n: int = 1000) -> dict:
    model, ck = load_checkpoint(model_path)
    if ck["arch"] != arch:
        raise SystemExit(f"{model_path} is a {ck['arch']} checkpoint; pass --arch {ck['arch']}")
    out = Path(out)
    t0 = time.time()
    if arch == "arc":
        _export(ArcExport(model), _dummy_arc(getattr(model, "haz_cols", 8)), out, ARC_INPUTS, ARC_OUTPUTS)
    else:
        _export(MLPExport(model), (torch.zeros(1, 80),), out, ["obs"], ["logits"])
    exp_s = time.time() - t0
    par = check_parity(out, model, arch, parity_frames(ck, parity_n))
    T, tau = 1.0, 0.0
    if calibration:
        cal = json.loads(Path(calibration).read_text())
        T, tau = float(cal["temperature"]), float(cal["tau"])
    summ_p = Path(model_path).parent / "summary.json"
    summ = json.loads(summ_p.read_text()) if summ_p.exists() else {}
    # the sensor spec the pilot reads: v2.3 tracker rows, or 8-value rows (v2.2, band rows) for older nets
    spec = C.SPEC_VERSION if getattr(model, "haz_cols", 8) == C.HAZ_COLS else "2.2"
    side = {"temperature": T, "tau": tau, "spec_version": spec, "val_ratio": summ.get("val_ratio"),
            "trained_from": str(model_path),
            # extras (not required by the SPEC; used by eval.report and the handoffs)
            "arch": arch, "stage": ck.get("stage"), "run": ck.get("run"), "val_level": summ.get("val_level"),
            "val_seeds": summ.get("val_seeds"), "calibration": calibration, "parity": par,
            "model_cfg": ck.get("model_cfg"), "exported": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.with_suffix(".json").write_text(json.dumps(side, indent=2))
    print(f"[export] {out} ({out.stat().st_size / 1024:.0f} KB, {exp_s:.1f} s) + {out.with_suffix('.json').name}")
    print(f"[export] parity on {par['frames']} frames ({par['source']}): max |logit diff| "
          f"{par['max_abs_diff_logits']:.2e}, other outputs {par['max_abs_diff_other_outputs']:.2e}, "
          f"mask mismatches {par['mask_mismatches']} -> {'PASS' if par['pass'] else 'FAIL'}")
    return side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m train.export_onnx", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="arc", choices=["arc", "mlp"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity-frames", type=int, default=1000)
    args = ap.parse_args(argv)
    side = export(args.model, args.out, args.arch, args.calibration, args.parity_frames)
    return 0 if side["parity"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
