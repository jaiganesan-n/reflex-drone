# arc: model card

`artifacts/arc.onnx` + `arc.json` (SPEC v2.3). Made 17 Oct 2026; chosen by the owner 18 Oct 2026.

## Inputs (one frame, 30 per second)

**Self, 11 values**

| # | Value |
|---|---|
| 0 | height / 10 |
| 1 | vertical speed / 8 |
| 2 | diving (tucked), 0/1 |
| 3 | JUMP/DUCK cooldown left / 1 s |
| 4 | (speed − 8) / 8 |
| 5 | felt wind (gust + downdraft) / 8 |
| 6–10 | last action, one-hot |

**Rays, 13 values:** first-hit distance / 16 at −60° … +60° every 10°. They reach 20 m (1.25); in fog
they reach 8 m (0.5), with noise. Hits are obstacles, the ground, the ceiling and storm clouds.

**Obstacles, up to 12 rows of 13 values**, nearest first, each with its kind and a valid flag:

| # | Value |
|---|---|
| 0 | gap from the nose to its near edge / 16 (up to 20 m) |
| 1–2 | its body now: bottom / 10, top / 10 |
| 3 | width / 8 |
| 4 | closing speed / 16 |
| 5 | vertical speed / 8 (birds and drones: from the tracker's last two sightings) |
| 6 | distance from the drone / 16 |
| 7 | moving, 0/1 |
| 8 | watched: seconds the tracker has followed it / 2, capped at 1 (fixed obstacles: 1) |
| 9–10 | lowest and highest body edge seen so far / 10 |
| 11–12 | predicted lowest and highest body edge while the drone passes it / 10 |

- **Kinds (9 in the game):** building, tree, street light, kite, storm cloud, bird, drone or balloon, plane, downdraft.
- **Fixed obstacles** are fully known: seen and predicted bands equal their shape.
- **Moving obstacles:** values 9–12 come from the object tracker (`drone_env/tracker.py`), which never reads the game's
  hidden path settings. It fits a sine to the last 1.5 s of sightings, evaluates it across the pass window, adds
  calibrated margins (`configs/tracker.yaml`) and keeps the band within what that kind of object can reach from what
  was seen.

**Derived inside the network, 5 per obstacle:**
- time to contact / 3 s;
- climb needed / 10 and descent needed / 10, measured to the predicted band;
- the slack of a plain ASCEND or of a plain DESCEND started now / 5.

These answer the question "is there time for a slow move, or does this need a burst?"

**Per action, 3 values:** height gained 0.5 s after starting it / 3, hitbox scale, and duration. Unavailable actions
(cooldown, inside a burst) are masked.

## Network: Arc (attention over a set of tokens)

| Part | Detail |
|---|---|
| Tokens | SELF, RAYS, 12 OBSTACLE, 5 ACTION, DECIDE; width d = 64; no position embeddings (a set) |
| Token encoders | Two-layer MLPs (GELU). SELF 11→64; RAYS 13→64; OBSTACLE 18→64 plus an embedding of its kind; ACTION: id embedding plus 3→64; DECIDE: a learned vector |
| Entity encoder | 2 transformer blocks (pre-norm, 4-head attention, feed-forward 64→256→64, GELU) with a structured mask: the context tokens see each other, each ACTION sees the context and itself, DECIDE sees everything valid |
| Choice head | 1 block among the 5 ACTION tokens and DECIDE |
| Outputs | **choice** logits (bilinear DECIDE · ACTION, unavailable actions −inf); **P(safe)** per action (still flying after 1 s); **threat** level 0–4; DECIDE's attention over the 12 obstacles (shown with `--show-attention`) |
| Size | 179,847 weights (179,782 in the ONNX file, without the training-only value head); ONNX 1 MB (opset 18); one forward pass per frame, `act()` median 0.14 ms on AC power |
| Decision | softmax(logits / T) with T = 0.32 (calibrated); "hesitating" when the top probability < 0.45 |

The feed-forward layers are 4 × d wide. The 2 × d version (130,311 weights) flew 7% less far on L3 VAL.

## Training method: imitation of a planner (System 2 → System 1)

1. **Teacher (System 2, `teacher/planner.py`, `configs/train/teacher.yaml`).** Beam search on calm copies of the game:
   - **Search:** macro actions of 5 frames, 9 deep (1.5 s), beam 2 per first action.
   - **Score:** frames survived, plus 0.5 × mean clearance (capped at 2 m), minus costs:
     - ASCEND or DESCEND: 0.05;
     - JUMP or DUCK: +0.95 more, plus 2.25 at 10 m/s or 1.5 at 12 m/s, so bursts are for sudden obstacles at speed.
   - **What it knows:** only what the drone's sensors show (20 m). Birds, drones and planes are planned as the
     tracker's predicted bands. If those leave no safe move, it plans against the tracker's best guess instead.
   - **Margins and pruning:** trees, street lights and kites are planned 0.2 m bigger; steady climbs are never pruned.
   - **Labels per frame:** soft targets softmax(scores / 2), "safe" per action (some plan survives 1 s), threat 0–4,
     and the chosen action.
2. **Data:**
   - **Size:** 602,038 frames (1,104 episodes) on TRAIN seeds.
   - **Level mix:** L1 10%, L2 15%, L3 20%, L4 25%, L5 30%.
   - **Exploration:** random 1–5 frame action bursts (15% chance per frame) show states off the teacher's own path.
3. **Supervised (run `arc_A`):**
   - **Schedule:** 15 epochs, AdamW (lr 3e-4, weight decay 0.01), cosine schedule with 5% warm-up, batch 512,
     gradient clip 1.0, Apple MPS.
   - **Loss:** KL(teacher ‖ arc) + 0.5 · BCE(safe per action) + 0.25 · CE(threat) + 0.5 · CE(teacher's action).
   - **Sampling:** dangerous frames (threat ≥ 2) and JUMP/DUCK frames are weighted 3×.
   - **Selection:** best epoch by a 10-seed L3 VAL check; result L3 VAL 2,009 m.
4. **Calibration:** temperature scaling on L5 VAL.
5. **DAgger (run `arc_B`):**
   - **Rounds:** 4 rounds of about 60k frames. Arc flies, with a 50%, 25%, 0%, 0% chance per round that the teacher's
     action is executed instead.
   - **Labels:** the teacher labels every state arc reaches, which teaches it to recover from its own mistakes.
   - **Retraining:** warm-started, 3 epochs per round, on all data so far (826k frames at the end).
   - **Selection:** best round (4) by L5 VAL.
6. **Calibrate again, export to ONNX** (parity check: logits within 1.2e-5 of PyTorch), evaluate.

arc is trained with DAgger and uses no reinforcement learning; arc_ppo (below) is arc fine-tuned with PPO. Both are
trained pilots and both fly in the game. At play time only the pilot runs; the planner is never called.

`scripts/train_arc.sh` runs steps 2–6.

## Results

| | L1 | L2 | L3 | L4 | L5 | stress |
|---|---|---|---|---|---|---|
| TEST (100 unseen seeds) | 1,947 m · 93% | 2,174 m · 77% | 2,020 m · 65% | 2,070 m · 63% | 1,697 m · 43% | 832 m · 2% |
| VAL (50 seeds) | 1,922 m · 92% | 2,112 m · 80% | 2,135 m · 76% | 1,881 m · 52% | 1,818 m · 48% | |

- **Distance:** mean distance and share of runs surviving 6,000 frames (200 s). L1 runs at 10 m/s, L2–L5 at 12, and
  stress at 14.4 with doubled interruptions.
- **Distance ratio** (against the reference planner, `teacher.yaml` with beam 2 and replan 5, on the same seeds):
  - L5 TEST 0.89 (planner 1,918 m, 57% survive). This is the headline result.
  - stress 0.98 (planner 851 m, 3% survive), so the stress drop is −0.10 (gate G6: ≤ 0.20).
  - L5 VAL 0.93 (planner 1,953 m).
  - The `val_ratio` 1.317 in `arc.json` and `runs/arc_B/summary.json` was measured against the earlier, plain planner
    (1,380 m on L5 VAL), before `teacher.yaml` became the reference.
- **Feel (8 VAL seeds):** holds 77–92% of the time; starts moves a median 1.0–1.15 s before an obstacle reaches its
  path; 0.3 JUMP/DUCK per min at L1 and 1.7 at L5.
- **Endless:** median 5.1 km in 18 long runs.
- **Known weakness:** needless bursts near the ceiling (13 of the 57 L5 TEST crashes and 22 of 98 in stress were the
  ceiling), and climbing into storm clouds over a bird.
- **Latency:** `act()` p50 0.14 ms, p99 0.16 ms on AC power (gate: p99 < 1 ms). The earlier 1.6 ms p99 was
  measured on battery in Low Power Mode.

## arc_ppo: arc fine-tuned with PPO

`artifacts/arc_ppo.onnx` + `arc_ppo.json` (run `arc_ppo`, 18 Oct 2026). Same inputs, network and size as arc; only
the weights differ.

**Training** (`train/ppo.py`, `configs/train/ppo.yaml`, `scripts/train_arc_ppo.sh`):

- **Start:** arc's weights (`runs/arc_B/best.pt`). 1,000 PPO updates, 8.19M game steps, about 3 h:
  - 8 games at once, 1,024 steps each per update, 4 epochs, minibatch 1,024;
  - lr 1e-4, clip 0.1, entropy 0.005, γ 0.99, GAE λ 0.95;
  - the first 50k steps train only the value head.
- **Policy:** sampled and updated as softmax(logits / 0.32), arc's calibrated temperature, so practice flies close to
  how arc plays.
- **Levels:** each practice run draws L1-L5 with weights 10/15/20/25/30%, on TRAIN seeds.
- **Reward:** the game's own (alive +0.01 per frame, obstacle cleared +1, crash −5, −0.002 per action change), plus
  training-only shaping:
  - clearance +0.02 per frame × min(clearance, 2 m) / 2 m (the ceiling counts);
  - −0.0167 per m/s of change in target climb rate;
  - −0.1 per JUMP or DUCK, plus 0.5 more at 12 m/s (0.75 at 10 m/s);
  - −1.0 more when the teacher finds that a slower move gets through the whole 1.5 s plan (a needless burst).
- **Anchors** (so practice keeps arc's feel):
  - + 0.5 × KL(policy ‖ arc) on the states it flies. This keeps arc's timing; earlier PPO pilots without it started
    moves too late.
  - + 0.5 × the supervised teacher losses on replayed teacher frames (no JUMP/DUCK targets where a slower move is
    safe).
- **Selection:** the best checkpoint by mean distance ratio on L1, L3 and L5 (10 VAL seeds each, every 20 updates):
  update 560. Then calibration (T 0.39, τ 0.45) and ONNX export (parity within 1.1e-5).

**Results** (the same seeds as arc; TEST and STRESS run once, 20 Oct 2026):

| | L1 | L2 | L3 | L4 | L5 | stress |
|---|---|---|---|---|---|---|
| arc_ppo, TEST (100 seeds) | 1,999 m · 98% | 2,071 m · 73% | 1,996 m · 64% | 2,035 m · 66% | 1,808 m · 51% | 852 m · 3% |
| arc, TEST | 1,947 m · 93% | 2,174 m · 77% | 2,020 m · 65% | 2,070 m · 63% | 1,697 m · 43% | 832 m · 2% |
| arc_ppo, VAL (50 seeds) | 2,012 m · 98% | 2,057 m · 72% | 2,092 m · 72% | 1,866 m · 56% | 1,937 m · 60% | |
| arc, VAL | 1,922 m · 92% | 2,112 m · 80% | 2,135 m · 76% | 1,881 m · 52% | 1,818 m · 48% | |

- **Distance:** a tie overall (L1–L5 TEST mean 1,982 m each; VAL 1,993 m vs 1,974 m). Better on L1 and L5, worse on
  L2 (TEST −103 ± 51 m), level in stress.
- **Distance ratio** (reference planner, same seeds): L5 TEST 0.94 (arc 0.89), stress 1.00 (arc 0.98), so the stress
  drop is −0.06 (gate G6: ≤ 0.20). L5 VAL 0.99 (planner 1,953 m).
- **Crashes:**
  - Ceiling: 5 vs 31 on L1–L5 TEST, 4 vs 22 in stress, 0 vs 15 on VAL.
  - More building and traffic-drone crashes: 34 vs 24 and 11 vs 7 on L1–L5 TEST, 25 vs 13 and 16 vs 8 in stress.
  - Storm clouds: 17 vs 10 on VAL, but level on TEST (20 vs 18) and fewer in stress (11 vs 26).
- **Feel:** like arc:
  - same holds and timing (moves start a median 1.0-1.13 s ahead) and same mean height;
  - a third fewer bursts (0.2 / 0.6 / 1.1 per min at L1 / L3 / L5).
- **Endless:** median 6.8 km in 18 long runs (arc 5.1 km, same seeds); 72% pass 6 km (arc 39%).
- **Latency:** `act()` p50 0.141 ms, p99 0.155 ms on AC power (gate: p99 < 1 ms).
- **In the game:** `uv run python -m app.play --agent onnx-arc:artifacts/arc_ppo.onnx`, the same way as arc. The owner
  played it in endless on 18 Oct 2026 (5,646 m and 5,214 m).
