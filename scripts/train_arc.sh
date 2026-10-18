#!/bin/bash
# Train arc from scratch (SPEC v2.3), from reflex-drone/: teacher data -> supervised (arc_A) -> DAgger (arc_B)
# -> calibration -> artifacts/arc.onnx -> VAL. About 1 h on an Apple Silicon MacBook on AC power (much slower on
# battery in Low Power Mode). Every step logs to the terminal; run it under `caffeinate -i`.
set -euo pipefail
cd "$(dirname "$0")/.."
t0=$(date +%s)
stamp() { echo "=== [$(date +%H:%M:%S), t+$(( $(date +%s) - t0 ))s] $*"; }
stamp "tests"
uv run python -m pytest -q -x -p no:cacheprovider
stamp "teacher data: data/teacher (600k frames)"
uv run python -m data.collect --config configs/train/collect.yaml --out data/teacher --target-frames 600000 --quiet
uv run python -m data.inspect data/teacher
stamp "supervised: arc_A"
uv run python -m train.supervised --config configs/train/arc.yaml --data data/teacher --run arc_A
uv run python -m train.calibrate --model runs/arc_A/best.pt --level L5 --seeds val --out runs/arc_A/calibration.json --teacher-config configs/train/teacher.yaml
stamp "DAgger: arc_B (4 rounds)"
uv run python -m data.dagger --config configs/train/dagger.yaml --init runs/arc_A/best.pt --calibration runs/arc_A/calibration.json --run arc_B
uv run python -m train.calibrate --model runs/arc_B/best.pt --level L5 --seeds val --out runs/arc_B/calibration.json --teacher-config configs/train/teacher.yaml
stamp "export: artifacts/arc.onnx"
uv run python -m train.export_onnx --model runs/arc_B/best.pt --calibration runs/arc_B/calibration.json --out artifacts/arc.onnx
stamp "VAL, all levels"
uv run python -m eval.evaluate --agent onnx-arc:artifacts/arc.onnx --levels L1 L2 L3 L4 L5 --seeds val --out reports/arc_val.json
stamp "done"
