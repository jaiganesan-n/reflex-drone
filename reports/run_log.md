# Run log

Every training or evaluation step, newest last. The log of everything before arc (sessions 1-3, game v1.4-v2.2,
pilots v19-g22) is in `archive/logs/reports/run_log.md`.

## arc becomes the only model (owner, 18 Oct 2026)

The owner played the g23 pilot ("g23 works well. Good.") and asked to erase the earlier commits, data and files of
earlier training, keeping them as a log outside git, and to rename g23 as the only model, arc.

- **Git:** the 52 earlier commits are saved as `archive/git_history_before_arc.bundle` (restore with
  `git clone archive/git_history_before_arc.bundle old-history`). Their text log is in
  `archive/logs/git_log_before_arc.txt`. The repository restarts with one commit of today's code.
- **Earlier training, as logs:** in `archive/logs/` (git-ignored):
  - run logs and summaries (`runs/`), session logs (`runs_logs/`), play-session logs (`play/`);
  - dataset metas (`data/`), model sidecars (`artifacts/`), reports and handoffs (`reports/`), training configs
    (`configs/`; the earlier evaluation reference teacher is `reference_teacher.yaml`).
- **Deleted (741 MB):** earlier datasets (g21, g22), checkpoints of 43 earlier runs, 21 earlier ONNX models
  (v19-v20, g2-g22, the MLP baselines, smoke).
- **Renamed:**
  - `artifacts/arc_g23_dagger.onnx` -> `artifacts/arc.onnx`;
  - runs `arc_g23_Aw` -> `arc_A`, `arc_g23_B` -> `arc_B`;
  - `data/teacher_g23` -> `data/teacher`;
  - configs `teacher_g23` / `collect_g23` / `arc_g23w` / `dagger_g23w` -> `teacher` / `collect` / `arc` / `dagger`.
  - Paths inside the sidecar, checkpoints, summaries and dataset meta were rewritten to match.
- **`teacher.yaml` is now arc's teacher.** It is also the evaluation reference; earlier planner caches went to the
  archive.
- **New:** `scripts/train_arc.sh` (the whole pipeline) and `reports/arc_model_card.md` (inputs, network, training,
  results). Tests 248 passed.

## arc: how it was made (17 Oct 2026, 22:41-23:45, AC power)

| Step | Result |
|---|---|
| teacher data `data/teacher` | 602,038 frames, 1,104 episodes, 691 s |
| supervised, FFN 2x d (130,311 weights; deleted) | L3 VAL 1,870 m, 30/50 survive |
| supervised `arc_A`, FFN 4x d (179,847 weights) | L3 VAL 2,009 m, 33/50 survive -> chosen (+7%) |
| DAgger `arc_B` (4 rounds, from `arc_A`) | best round 4; 826,339 frames at the end |
| calibration (L5 VAL) | T 0.32, tau 0.45, agreement with the teacher 96.5% |
| export `artifacts/arc.onnx` | parity: logits within 1.2e-5 of PyTorch |

## arc: VAL, TEST and STRESS (TEST/STRESS once, after the owner chose it)

| | L1 | L2 | L3 | L4 | L5 | stress |
|---|---|---|---|---|---|---|
| VAL, 50 seeds (`reports/arc_val.json`) | 1,922 m · 92% | 2,112 m · 80% | 2,135 m · 76% | 1,881 m · 52% | 1,818 m · 48% | |
| TEST, 100 seeds (`reports/arc_final.json`) | 1,947 m · 93% | 2,174 m · 77% | 2,020 m · 65% | 2,070 m · 63% | 1,697 m · 43% | 832 m · 2% |

- **TEST crashes:**
  - L1: tree 4, ceiling, building, kite.
  - L3: tree 12, building 5, ceiling 5, pole 4, storm 4.
  - L5: ceiling 13, bird 11, storm 9, building 8, traffic 5.
  - stress (STRESS seeds, 14.4 m/s, doubled interruptions, gust + fog + surge at once): storm 26, ceiling 22, bird 13,
    building 13.
- **Latency** (`reports/latency_arc.json`, 10,000 calls, one thread, on battery): `act()` p50 0.317 ms, p99 1.607 ms
  (gate G6: p99 < 1 ms, fail); `session.run` p50 0.369 ms, p99 0.805 ms. Re-measure on AC power.
- **Feel and endless** (VAL): holds 77-92% of the time; moves start a median 1.03 / 1.13 / 1.15 / 1.04 s ahead
  (L1 / L3 / L5 / endless); 0.3 / 1.1 / 1.7 / 0.3 bursts per min; endless median 5.1 km in 18 long runs.

## arc_ppo: PPO fine-tuning of arc (owner, 18 Oct 2026)

The owner: "Name the current model as arc, fine tune with PPO (model name arc_ppo). We can compare both models."
arc keeps its name (`artifacts/arc.onnx`); arc_ppo is a second pilot, compared with arc on VAL seeds. TEST and
STRESS for arc_ppo run only if the owner chooses it.

- **Code** (commit `41f8a38`): `ref_kl_coef` in `train/ppo.py`, + coef × KL(policy ‖ starting policy) on the states
  PPO flies. Earlier PPO runs (archive) started moves later (1.0-1.6 s → 0.5-1.0 s ahead) even with a strong teacher
  anchor, because the teacher's soft labels barely tell "climb now" from "hold a little longer".
- **Config** (commit `c565ee9`, `configs/train/ppo.yaml`):
  - from `runs/arc_B/best.pt`, 1,000 updates of 8 × 1,024 steps;
  - policy temperature 0.32 (arc's), KL anchor 0.5, aux 0.5 on `data/teacher`;
  - level mix L1-L5 10/15/20/25/30%;
  - shaping: clearance 0.02, target change 0.0167, burst 0.1, slow burst 1.0, needless burst 1.0 (1.5 s check);
  - no vertical-speed cost and no clear-path bonus: they made earlier PPO pilots late and high;
  - best.pt = best mean online ratio on L1, L3, L5 (10 VAL seeds each) from update 100.
- **Run:** `bash scripts/train_arc_ppo.sh` (PPO → calibrate → `artifacts/arc_ppo.onnx` → VAL), then the comparison
  (feel, snapshots, endless, latency). Log `runs/logs/arc_ppo.log`. Started 18 Oct 2026 on battery (76%, Low Power
  Mode), under the battery guard.

### arc_ppo results (18 Oct 2026, 11:19-14:39; on battery in Low Power Mode until ~12:50, then AC)

| Step | Result |
|---|---|
| PPO `arc_ppo` | 1,000 updates, 8.19M steps in 190 min (480-600 steps/s on battery, ~1,300 on AC). Online mean ratio (L1/L3/L5, 10 VAL seeds each): start 1.128, then 0.93-1.23 with no trend. Practice flights: bursts 7.3 → ~1/min, needless share 80% → ~30%, HOLD ~70%, KL to arc 0.01-0.025 throughout. **best.pt = update 560** (online 1.230); 50 L5 VAL seeds: 1,937 m, 30 survived (arc 1,818 m, 24) |
| calibration (L5 VAL) | T 0.387 (arc 0.321), tau 0.45, agreement with the teacher 96.4% (arc 96.5%) |
| export `artifacts/arc_ppo.onnx` | parity: logits within 1.1e-5 of PyTorch |

**VAL, 50 seeds per level, paired with arc** (`reports/arc_ppo_val.json`, `reports/arc_val.json`):

| | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| arc | 1,922 m · 92% | 2,112 m · 80% | 2,135 m · 76% | 1,881 m · 52% | 1,818 m · 48% |
| arc_ppo | 2,012 m · 98% | 2,057 m · 72% | 2,092 m · 72% | 1,866 m · 56% | 1,937 m · 60% |
| paired difference ± 1 se | +91 ± 54 m | −55 ± 130 m | −43 ± 92 m | −15 ± 133 m | +120 ± 130 m |
| seeds 10-49 only (not used to pick best.pt) | +92 ± 64 m | −94 ± 134 m | −65 ± 115 m | −48 ± 150 m | +76 ± 142 m |

- **Crashes (all VAL levels):**
  - arc: building 17, tree 16, ceiling 15, storm 10, bird 8, pole 5, kite 3, traffic 2.
  - arc_ppo: storm 17, building 12, tree 12, bird 10, traffic 7, pole 6, kite 3, ground 3, plane 1.
  - **No ceiling crashes** (arc: 15); more storm and traffic.
- **Feel (8 VAL seeds; `pilot_feel.py`):** arc_ppo flies like arc, unlike every earlier PPO pilot:
  - same holds (92 / 84 / 76 / 91% at L1 / L3 / L5 / endless);
  - same timing: moves start a median 1.04 / 1.13 / 1.12 / 0.99 s ahead (arc 1.03 / 1.13 / 1.15 / 1.04);
  - same mean height (6.6 / 6.1 / 5.8 / 6.5 m);
  - fewer bursts: 0.2 / 0.6 / 1.1 / 0.2 per min (arc 0.3 / 1.1 / 1.7 / 0.3).
  - The snapshots every 100 updates kept 1.08-1.18 s and 5.6-6.3 m throughout: the KL anchor held the timing.
- **Endless, 18 long runs** (same seeds, same script, both on AC):
  - arc_ppo: median 6,754 m, 72% past 6 km.
  - arc: median 5,086 m, 39% past 6 km.
  - Paired: arc_ppo +1,613 ± 832 m per run, farther on 12 of 18.
  - arc_ppo's runs ended at storm 4, traffic 3, pole 3, building 2, tree 2, kite 2, plane 1, ground 1.
- **Latency on AC** (10,000 calls): arc `act()` p50 0.142 ms, p99 0.160 ms; arc_ppo p50 0.141 ms, p99 0.155 ms. G6
  passes for both. The 1.6 ms p99 measured earlier on battery came from Low Power Mode.
- **Reading:**
  - Mean distance over L1-L5 is a tie (1,993 m vs 1,974 m); only the L1 gain is beyond the noise.
  - arc_ppo is steadier: no ceiling crashes, a third fewer bursts, and longer endless runs.
  - It is weaker on storm clouds and traffic drones that come suddenly.
  - TEST and STRESS for arc_ppo wait for the owner's choice.
