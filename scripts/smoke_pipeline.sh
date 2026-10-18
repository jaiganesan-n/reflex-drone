#!/usr/bin/env bash
# Smoke pipeline: every SPEC §15 step at toy size (SPEC §14 isolation).
#
#   bash scripts/smoke_pipeline.sh            # all steps (cleans the smoke outputs first)
#   bash scripts/smoke_pipeline.sh ppo export # only some steps, in pipeline order
#
# Writes only to data/smoke/, runs/smoke_*, reports/smoke/ and artifacts/smoke_*, and only uses
# VAL seeds. Never touches real data, the real planner caches, TEST or STRESS seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

UV=${UV_RUN:-"uv run"}
PY="$UV python -m"
STEPS=(clean collect inspect overfit arc mlp calibrate dagger ppo export evaluate latency report)
if [ $# -gt 0 ]; then WANT=" $* "; else WANT=" ${STEPS[*]} "; fi
T0=$(date +%s)
step() { [[ "$WANT" == *" $1 "* ]]; }
banner() { echo; echo "=== [smoke] $1  (t+$(( $(date +%s) - T0 ))s) ==="; }

if step clean; then
  banner "clean smoke outputs"
  rm -rf data/smoke reports/smoke runs/smoke_* artifacts/smoke_*
fi
mkdir -p reports/smoke artifacts

if step collect; then
  banner "collect ~600 frames (VAL seeds)"
  $PY data.collect --config configs/train/smoke_collect.yaml --out data/smoke
fi
if step inspect; then
  banner "dataset card"
  $PY data.inspect data/smoke > /dev/null
  echo "data/smoke/CARD.md written"
fi
if step overfit; then
  banner "overfit 10 frames (loss must fall below 0.05)"
  $PY train.supervised --config configs/train/smoke.yaml --data data/smoke --run smoke_overfit
fi
if step arc; then
  banner "1 epoch Arc"
  $PY train.supervised --config configs/train/smoke_arc.yaml --data data/smoke --run smoke_arc
fi
if step mlp; then
  banner "1 epoch MLP"
  $PY train.supervised --config configs/train/smoke_mlp.yaml --data data/smoke --run smoke_mlp
fi
if step calibrate; then
  banner "calibrate (500 frames)"
  $PY train.calibrate --model runs/smoke_arc/best.pt --level L5 --seeds val \
      --out runs/smoke_arc/calibration.json --max-frames 500 --workers 2
fi
if step dagger; then
  banner "DAgger: 1 round of 200 frames"
  $PY data.dagger --config configs/train/smoke_dagger.yaml --init runs/smoke_arc/best.pt \
      --calibration runs/smoke_arc/calibration.json --run smoke_dagger
fi
if step ppo; then
  banner "PPO: 2-minute budget"
  $PY train.ppo --config configs/train/smoke_ppo.yaml --init runs/smoke_dagger/best.pt --run smoke_ppo \
      --time-budget-min 2
fi
if step export; then
  banner "export ONNX + sidecars"
  $PY train.export_onnx --model runs/smoke_ppo/best.pt --calibration runs/smoke_arc/calibration.json \
      --out artifacts/smoke_arc.onnx
  $PY train.export_onnx --model runs/smoke_arc/best.pt --calibration runs/smoke_arc/calibration.json \
      --out artifacts/smoke_arc_supervised_only.onnx
  $PY train.export_onnx --arch mlp --model runs/smoke_mlp/best.pt --out artifacts/smoke_mlp.onnx
fi
if step evaluate; then
  banner "2-seed VAL evaluation (600-frame cap)"
  EV="--levels L3 L5 --seeds val --max-seeds 2 --workers 2 --cache-dir reports/smoke --frame-cap 600"
  $PY eval.evaluate --agent planner $EV --out reports/smoke/planner_val.json
  $PY eval.evaluate --agent onnx-arc:artifacts/smoke_arc.onnx $EV --out reports/smoke/arc_val.json
  $PY eval.evaluate --agent onnx-arc:artifacts/smoke_arc_supervised_only.onnx $EV --out reports/smoke/arc_supervised_only_val.json
  $PY eval.evaluate --agent onnx-mlp:artifacts/smoke_mlp.onnx $EV --out reports/smoke/mlp_val.json
fi
if step latency; then
  banner "latency (10,000 calls, one thread)"
  $PY eval.latency --agent onnx-arc:artifacts/smoke_arc.onnx --calls 10000 --eager --out reports/smoke/latency_smoke_arc.json
  $PY eval.latency --agent onnx-mlp:artifacts/smoke_mlp.onnx --calls 10000 --out reports/smoke/latency_smoke_mlp.json
fi
if step report; then
  banner "report"
  $PY eval.report reports/smoke/planner_val.json reports/smoke/arc_val.json \
      reports/smoke/arc_supervised_only_val.json reports/smoke/mlp_val.json --out reports/smoke/results.md > /dev/null
  echo "reports/smoke/results.md written"
fi
banner "done in $(( $(date +%s) - T0 )) s"
