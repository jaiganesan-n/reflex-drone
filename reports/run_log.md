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
