#!/bin/bash
# Fine-tune arc with PPO (configs/train/ppo.yaml), from reflex-drone/: PPO from runs/arc_B/best.pt (arc_ppo)
# -> calibration -> artifacts/arc_ppo.onnx -> VAL. About 2.5 h on an Apple Silicon MacBook on AC power (about twice
# that on battery in Low Power Mode). Every step logs to the terminal; run it under `caffeinate -i`.
set -euo pipefail
cd "$(dirname "$0")/.."
t0=$(date +%s)
stamp() { echo "=== [$(date +%H:%M:%S), t+$(( $(date +%s) - t0 ))s] $*"; }
stamp "tests"
uv run python -m pytest -q -x -p no:cacheprovider
stamp "PPO: arc_ppo (from arc, 1,000 updates)"
uv run python -m train.ppo --config configs/train/ppo.yaml --init runs/arc_B/best.pt --run arc_ppo
stamp "calibrate arc_ppo"
uv run python -m train.calibrate --model runs/arc_ppo/best.pt --level L5 --seeds val --out runs/arc_ppo/calibration.json --teacher-config configs/train/teacher.yaml
stamp "export: artifacts/arc_ppo.onnx"
uv run python -m train.export_onnx --model runs/arc_ppo/best.pt --calibration runs/arc_ppo/calibration.json --out artifacts/arc_ppo.onnx
stamp "VAL, all levels"
uv run python -m eval.evaluate --agent onnx-arc:artifacts/arc_ppo.onnx --levels L1 L2 L3 L4 L5 --seeds val --out reports/arc_ppo_val.json
stamp "done"
