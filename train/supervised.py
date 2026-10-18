"""Supervised training on teacher labels (SPEC §13, §15).

    uv run python -m train.supervised --config configs/train/arc.yaml --data data/teacher --run arc_A

Loss: KL(p_teacher || p) + 0.5 * sum BCE(safe) + 0.25 * CE(threat), masked actions excluded
(MLP: KL only). AdamW (lr 3e-4, wd 0.01), cosine schedule with 5% warm-up, grad clip 1.0,
batch 512, weighted sampling (threat >= 2 x3; hesitant DAgger frames x2; teacher JUMP/DUCK frames
x burst_weight, default 1).

Every epoch: offline metrics on the 5% holdout episodes (KL, teacher agreement, safe AUC, threat
accuracy) and an online check: fly the model on 10 VAL seeds of the gate level and compute the
distance ratio against the planner cache. best.pt = best online ratio. At the end best.pt is
flown on all 50 VAL seeds of the gate level and summary.json gets val_ratio / val_level /
val_seeds (the number the gates read).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from data.dataset import FrameData
from models import build_model, load_checkpoint, policy_outputs, save_checkpoint
from train.common import (SUPERVISED_DEFAULTS, Run, as_list, contract_loss, final_val, load_config, lr_lambda,
                          offline_metrics, online_eval, select_device, set_threads)


def _label_temperature(data_dirs) -> float:
    from data.dataset import read_meta
    for d in data_dirs:
        m = read_meta(d) or {}
        if "teacher" in m:
            return float(m["teacher"].get("label_temperature", 2.0))
    return 2.0


def _bench_fn(model, data, arch, batch_size):
    import copy

    def bench(dev):
        m = copy.deepcopy(model).to(dev).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        idx = data.train_idx[:batch_size]
        b = data.batch(idx, dev)
        times = []
        for i in range(6):
            t0 = time.perf_counter()
            out = policy_outputs(m, arch, b)
            loss, _ = contract_loss(out, b, arch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if dev.type == "mps":
                torch.mps.synchronize()
            if i >= 2:
                times.append(time.perf_counter() - t0)
        return float(np.median(times))
    return bench


def train_supervised(cfg: dict, data_dirs, run_name: str, init: str | None = None, tau: float | None = None,
                     run_root: str | Path | None = None, do_final_val: bool = True, quiet: bool = False,
                     model_override=None) -> dict:
    data_dirs = as_list(data_dirs)
    cfg = dict(cfg, data=data_dirs, init=init, tau=tau)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    set_threads(int(cfg["threads"]))
    run = Run(run_name, cfg, root=run_root)
    arch = cfg["arch"]
    data = FrameData(data_dirs, tau=tau, mlp=(arch == "mlp"), label_temperature=_label_temperature(data_dirs),
                     burst_weight=float(cfg["burst_weight"]))
    train_idx = data.train_idx
    if cfg["overfit_frames"]:
        g = torch.Generator().manual_seed(int(cfg["seed"]))
        train_idx = train_idx[torch.randperm(len(train_idx), generator=g)[: int(cfg["overfit_frames"])]]
        data.holdout_idx = train_idx
    # model (warm start keeps the checkpoint's architecture)
    if init:
        model, ck = load_checkpoint(init)
        if ck["arch"] != arch:
            raise SystemExit(f"--init {init} is a {ck['arch']} checkpoint but the config says arch={arch}")
        cfg["model"] = dict(ck["model_cfg"])
        model.train()
    else:
        model = model_override if model_override is not None else build_model(arch, cfg["model"])
    bs = int(cfg["batch_size"])
    device = select_device(cfg["device"], _bench_fn(model, data, arch, min(bs, len(train_idx))))
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    n_train = len(train_idx)
    if cfg["overfit_frames"]:
        steps_per_epoch = 1
        bs = n_train
    else:
        steps_per_epoch = int(cfg["steps_per_epoch"]) or math.ceil(n_train / bs)
    epochs = int(cfg["epochs"])
    total = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, float(cfg["warmup_frac"]), cfg["schedule"]))
    log_every = max(1, int(cfg.get("log_every", 1) or 1)) if "log_every" in cfg else (50 if cfg["overfit_frames"] else 1)
    if not quiet:
        print(f"[train] run={run.dir} arch={arch} params={sum(p.numel() for p in model.parameters()):,} "
              f"device={device} train={n_train:,} holdout={len(data.holdout_idx):,} frames "
              f"epochs={epochs} steps/epoch={steps_per_epoch} batch={bs}", flush=True)
    best_ratio, best_epoch = -math.inf, None
    train_seconds = 0.0
    gen = torch.Generator().manual_seed(int(cfg["seed"]) + 1)
    step = 0
    last_loss = None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if cfg["overfit_frames"]:
            order = train_idx
        else:
            order = data.epoch_indices(gen) if len(train_idx) == len(data.train_idx) else train_idx
            if steps_per_epoch * bs < len(order):
                order = order[: steps_per_epoch * bs]
        sums = {}
        nb = 0
        for a in range(0, len(order), bs):
            b = data.batch(order[a:a + bs], device)
            out = policy_outputs(model, arch, b)
            loss, parts = contract_loss(out, b, arch, cfg["loss"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            sched.step()
            step += 1
            nb += 1
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
        if device.type == "mps":
            torch.mps.synchronize()
        ep_s = time.time() - t0
        train_seconds += ep_s
        train_m = {f"train_{k}": v / max(nb, 1) for k, v in sums.items()}
        last_loss = train_m.get("train_loss")
        if epoch % log_every and epoch != epochs:
            continue
        row = {"epoch": epoch, "step": step, "lr": sched.get_last_lr()[0], "epoch_seconds": round(ep_s, 2),
               "train_frames_per_s": round(nb * bs / max(ep_s, 1e-9), 1), **train_m}
        row.update({f"holdout_{k}": v for k, v in offline_metrics(model, arch, data, device,
                                                                     int(cfg["holdout_max_frames"])).items()})
        save_checkpoint(run.path("last.pt"), model, arch, stage=cfg["stage"], run=run_name, epoch=epoch,
                        config=cfg)
        if cfg["online_eval"]:
            on = online_eval(model, arch, cfg["gate_level"], int(cfg["online_seeds"]), int(cfg["eval_frame_cap"]),
                             cfg["planner_cache_dir"], int(cfg["workers"]), device)
            row.update({f"online_{k}": v for k, v in on.items()})
            if on["ratio"] > best_ratio:
                best_ratio, best_epoch = on["ratio"], epoch
                save_checkpoint(run.path("best.pt"), model, arch, stage=cfg["stage"], run=run_name, epoch=epoch,
                                config=cfg, online_ratio=on["ratio"])
        run.log(row)
        if not quiet:
            msg = (f"[train] epoch {epoch}/{epochs}  loss {row.get('train_loss', float('nan')):.4f}  "
                   f"holdout kl {row['holdout_kl']:.4f} agree {row['holdout_agreement']:.3f}")
            if "holdout_safe_auc" in row:
                msg += f" safeAUC {row['holdout_safe_auc']:.3f} threat {row['holdout_threat_acc']:.3f}"
            if cfg["online_eval"]:
                msg += (f"  |  online {cfg['gate_level']} ratio {row['online_ratio']:.3f} "
                        f"({row['online_mean_distance_m']:.0f} / {row['online_planner_mean_distance_m']:.0f} m)")
            print(msg + f"  [{ep_s:.1f} s]", flush=True)
    if not cfg["online_eval"] or best_epoch is None:
        save_checkpoint(run.path("best.pt"), model, arch, stage=cfg["stage"], run=run_name, epoch=epochs, config=cfg)
        best_epoch = epochs
    summary = {"run": run_name, "arch": arch, "stage": cfg["stage"], "data": data_dirs, "init": init,
               "best_epoch": best_epoch, "best_online_ratio": None if best_ratio == -math.inf else best_ratio,
               "epochs": epochs, "steps": step, "train_frames": n_train, "holdout_frames": int(len(data.holdout_idx)),
               "device": str(device), "train_seconds": round(train_seconds, 1),
               "seconds_per_epoch": round(train_seconds / max(epochs, 1), 2), "final_train_loss": last_loss}
    if cfg["overfit_frames"]:
        model.eval()
        with torch.no_grad():
            b = data.batch(train_idx, device)
            fl, _ = contract_loss(policy_outputs(model, arch, b), b, arch, cfg["loss"])
        summary["overfit_loss"] = float(fl)
    if do_final_val and int(cfg["final_val_seeds"]) > 0:
        summary.update(final_val(run.path("best.pt"), cfg["gate_level"], int(cfg["final_val_seeds"]),
                                 int(cfg["eval_frame_cap"]), cfg["planner_cache_dir"], int(cfg["workers"])))
        if not quiet:
            print(f"[train] best.pt (epoch {best_epoch}) on {summary['val_seeds']} VAL seeds of "
                  f"{summary['val_level']}: val_ratio {summary['val_ratio']:.3f} "
                  f"({summary['val_mean_distance_m']:.0f} / {summary['val_planner_mean_distance_m']:.0f} m)", flush=True)
    run.write_summary(summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m train.supervised", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", nargs="+", default=None, help="dataset dir(s); default: the config's data")
    ap.add_argument("--run", required=True, help="run name: writes runs/<run>/")
    ap.add_argument("--init", default=None, help="warm start from a checkpoint")
    ap.add_argument("--epochs", type=int, default=None, help="override the config's epochs")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, SUPERVISED_DEFAULTS)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    s = train_supervised(cfg, args.data or cfg["data"], args.run, init=args.init or cfg.get("init"))
    thr = cfg.get("assert_loss_below")
    if thr is not None:
        val = s.get("overfit_loss", s.get("final_train_loss"))
        ok = val is not None and val < float(thr)
        print(f"[train] overfit check: loss {val:.4f} {'<' if ok else '>='} {thr}  ->  {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
