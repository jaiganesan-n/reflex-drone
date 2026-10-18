# Reflex Drone · DroneDash

A Chrome-Dino-style side-scroller where a drone flies itself. Every frame (30 Hz) the pilot sees
only its sensors and picks one of five actions: HOLD, ASCEND, DESCEND, JUMP, DUCK. The pilot is
**arc**, a small "System 1" network (Project J'AI 2), or **arc_ppo**, arc fine-tuned with PPO. A person can only **Start, Pause/Resume and
End** a run; there are no flying controls.

The contract is `../docs/SPEC.md` (v2.3). The run log is `reports/run_log.md`.

## Install

Needs [uv](https://docs.astral.sh/uv/) (it fetches Python 3.11 if you don't have it).

```bash
cd reflex-drone
uv sync
```

## Play

```bash
uv run python -m app.play --agent onnx-arc:artifacts/arc.onnx                                  # arc, endless mode
uv run python -m app.play --agent onnx-arc:artifacts/arc.onnx --level L5 --seed 42 --show-attention
uv run python -m app.play --agent onnx-arc:artifacts/arc_ppo.onnx                              # arc_ppo (PPO fine-tune of arc)
uv run python -m app.play --agent rule                                                         # hand-written test pilot
```

| Key / button | What it does |
|---|---|
| Enter, S, **Start** | Start a run (fresh random seed unless `--seed` was given) |
| P, Space, **Pause** | Pause / resume. The simulation and the agent freeze. |
| Q, Esc, **End** | End the run (or, on the summary screen, back to the title) |
| Close the window | Quit |

Flags (fixed for the whole session): `--agent` (`onnx-arc:<path>`, `rule`, `random`), `--level`
(`L1`–`L5`, `endless` (default), `stress`), `--seed N`, `--log-frames`, `--show-attention`
(outline the obstacle arc attends to). Each run writes `runs/play/<YYYYmmdd-HHMMSS>_seed<seed>.json`
(plus a per-frame CSV with `--log-frames`).

## The game (SPEC v2.3)

Real-life obstacles for a drone flying along a road: buildings, trees, street lights, kites,
storm clouds, birds, other drones and hot-air balloons, planes (high up), and downdrafts; plus
gusts, speed surges and fog.

| Level | Speed | Obstacles |
|---|---|---|
| L1 | 10 m/s | sparse, no combos |
| L2–L5 | 12 m/s | each level denser, more birds and drones, more combos (two hazards at one spot), more interruptions |

`endless` (the default) starts at L1 and moves up a level every 1.75 km (10 → 12 m/s over L1);
past 7 km the HUD shows **L5+** and obstacles keep getting denser. There is no finish line.
Every obstacle passes the generator's solvability checks before it is placed.

- **Birds and drones are not fully predictable.** Birds swing up and down with a rhythm that wanders,
  and some make a short sudden dive or pop-up; drones and balloons drift with a wandering rhythm.
- **Sensors reach 20 m** (8 m in fog): 13 rays plus a list of up to 12 nearby obstacles. The window
  (26 m wide, 1248 × 600) shows everything the pilot sees.
- **Object tracker.** Like a real drone's sensing software, the drone knows only what it has seen
  of each bird or drone: how long it has watched it, the lowest and highest heights seen, and a
  predicted band where it can be while the drone passes it. The band is wide at first sight and
  narrows as the tracker learns the object's rhythm (`drone_env/tracker.py`).
- **JUMP and DUCK are bursts** (±6 m/s for 0.3 s, a shared 1 s cooldown) for sudden obstacles at
  speed; ASCEND and DESCEND (3 m/s) are the normal moves.

## The pilots: arc and arc_ppo

`artifacts/arc.onnx` (runs `arc_A` → `arc_B`): imitation of a planner, then DAgger.
`artifacts/arc_ppo.onnx` (run `arc_ppo`): arc fine-tuned with PPO in the game. Same inputs and network.
Both fly on their own; the teacher (planner) is never used at play time.

| VAL, 50 seeds | L1 | L2 | L3 | L4 | L5 | endless (18 runs, median) |
|---|---|---|---|---|---|---|
| arc | 1,922 m · 92% | 2,112 m · 80% | 2,135 m · 76% | 1,881 m · 52% | 1,818 m · 48% | 5.1 km |
| arc_ppo | 2,012 m · 98% | 2,057 m · 72% | 2,092 m · 72% | 1,866 m · 56% | 1,937 m · 60% | 6.8 km |

Mean distance and share of runs that survive the 6,000-frame cap (200 s).

- **arc on TEST** (100 unseen seeds, once, after it was chosen):
  - L1 1,947 m · 93%; L2 2,174 m · 77%; L3 2,020 m · 65%; L4 2,070 m · 63%; L5 1,697 m · 43%;
  - stress 832 m · 2%.
- **Feel:** both hold 76–92% of the time and start their moves a median 1.0–1.15 s before an obstacle reaches
  their path. Mean height is 5.8–6.6 m for both.
- **Bursts:** arc JUMPs or DUCKs 0.3 per min at L1 and 1.7 at L5; arc_ppo 0.2 and 1.1.
- **Differences:** arc's weakness is needless bursts near the ceiling (15 ceiling crashes on VAL); arc_ppo has none.
  arc_ppo is weaker against sudden storm clouds and traffic drones.
- **Speed:** `act()` median 0.14 ms, p99 0.16 ms (on AC power).

Inputs, network and training: `reports/arc_model_card.md`.

## Training (about 1 h on AC power)

```bash
bash scripts/train_arc.sh        # teacher data -> supervised -> DAgger -> calibrate -> artifacts/arc.onnx -> VAL
bash scripts/train_arc_ppo.sh    # PPO from arc -> calibrate -> artifacts/arc_ppo.onnx -> VAL (~2.5 h on AC)
bash scripts/smoke_pipeline.sh   # every step at toy size (~3 min)
uv run python -m eval.evaluate --agent onnx-arc:artifacts/arc.onnx --levels L1 L2 L3 L4 L5 stress --seeds test --out reports/arc_final.json   # once, final only
uv run python -m eval.latency --agent onnx-arc:artifacts/arc.onnx --calls 10000 --out reports/latency_arc.json
uv run python scripts/calibrate_tracker.py   # re-calibrate the tracker's margins after changing bird or drone motion
```

Settings: `configs/train/teacher.yaml` (the teacher), `collect.yaml`, `arc.yaml`, `dagger.yaml`, `ppo.yaml` (arc_ppo).
`runs/`, `data/*/`, `artifacts/` and `archive/` are git-ignored: the trained files exist only on
the machine that made them. `archive/` holds the logs of earlier training and a bundle of the
earlier git history (`git clone archive/git_history_before_arc.bundle old-history`).

## Test and benchmark

```bash
uv run pytest -q                        # about 1.5 min
uv run python -m drone_env.bench        # simulator speed (one core)
uv run python -m app.rule_eval          # rule agent on the 50 L1 VAL seeds
```

## Use it from code

```python
import gymnasium, drone_env
env = gymnasium.make("DroneDash-v0", level="L3", max_frames=10_000)
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(0)

from drone_env.sim import Simulator           # planners only (never an agent)
sim = Simulator("L5", seed=7)
c = sim.clone(calm=True)
crashed = c.step(1)

from app.agents_base import Decision, load_agent, register_agent
```

## Layout

```
configs/      game.yaml (physics, hazards, generator, sensors) · levels.yaml · tracker.yaml (calibrated margins) · train/
drone_env/    constants, sim, generator, events, sensors, tracker, env, wrappers, render, seeds, bench
app/          play (app loop) · controls (Start/Pause/End) · hud · agents_base (rule, random) · rule_eval
teacher/      planner (System 2 beam search) · parallel
data/         collect · rollout · dataset · inspect · dagger
models/       arc (System 1) · mlp (flat baseline) · policy
train/        supervised · calibrate · ppo (arc_ppo) · export_onnx · common
eval/         evaluate · latency · report · planner_cache
agents/       onnx_agents (onnx-arc, onnx-mlp)
scripts/      train_arc.sh · train_arc_ppo.sh · smoke_pipeline.sh · calibrate_tracker.py
tests/        env/ app/ teacher/ models/ train/
reports/      run_log.md · arc_model_card.md · arc_val.json / arc_ppo_val.json / arc_final.json (git-ignored JSON)
```
