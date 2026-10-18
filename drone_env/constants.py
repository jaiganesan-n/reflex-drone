"""Constants for DroneDash, loaded once from configs/game.yaml and configs/levels.yaml.

Everything the simulator needs is exposed as plain module-level floats/ints so the hot
loop does attribute lookups on a module, not dict lookups on nested YAML.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import yaml

SPEC_VERSION = "2.3"   # v1.4: session 2 teacher (§11); v1.5: speed-based levels (§2-3); v1.6: display (§9);
                       # v1.7: +2 m/s, planes, DUCK = dive, endless L5+ density ramp (§2-4);
                       # v1.8: System 1 model renamed to Arc (rename only); v1.9: planes fly high (§3);
                       # v2.0: real-life obstacles: trees, street lights, power lines, cranes, kites;
                       #       mountains and planes retired; more traffic on L1-L2 (§3)
                       # v2.1: power lines and cranes retired, planes back, traffic drifts (§3)
                       # v2.2: a bobbing mover's hazard row gives its whole band (§5)
                       # v2.3: L1 10 / L2-L5 12 m/s, 20 m sensors, less predictable birds, object
                       #       tracker: 13-value hazard rows (§3, §5)
SENSOR_SPEC = (2, 3)   # what observe() builds unless a sim is told otherwise (older pilots)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
GAME_YAML = CONFIG_DIR / "game.yaml"
LEVELS_YAML = CONFIG_DIR / "levels.yaml"
TRACKER_YAML = CONFIG_DIR / "tracker.yaml"   # v2.3 calibrated tracker margins

with open(GAME_YAML) as f:
    GAME = yaml.safe_load(f)
with open(LEVELS_YAML) as f:
    LEVELS_CFG = yaml.safe_load(f)


def config_hash() -> str:
    """sha256 over game.yaml + levels.yaml (+ tracker.yaml, v2.3) bytes. Planner caches and dataset
    meta record it."""
    h = hashlib.sha256()
    h.update(GAME_YAML.read_bytes())
    h.update(b"\0")
    h.update(LEVELS_YAML.read_bytes())
    if TRACKER_YAML.exists():
        h.update(b"\0")
        h.update(TRACKER_YAML.read_bytes())
    return h.hexdigest()


# --- time ---------------------------------------------------------------------------
FPS: int = int(GAME["sim"]["fps"])
DT: float = 1.0 / FPS

# --- world --------------------------------------------------------------------------
WORLD_W: float = float(GAME["world"]["width_m"])
WORLD_H: float = float(GAME["world"]["height_m"])
CEILING: float = WORLD_H
DRONE_X: float = float(GAME["world"]["drone_x_m"])

# --- drone --------------------------------------------------------------------------
HIT_W: float = float(GAME["drone"]["hitbox_w"])
HIT_H: float = float(GAME["drone"]["hitbox_h"])
HALF_W: float = HIT_W / 2
HALF_H: float = HIT_H / 2
TUCK_SCALE: float = float(GAME["drone"]["tuck_scale"])
HALF_H_TUCK: float = HALF_H * TUCK_SCALE
START_Y: float = float(GAME["drone"]["start_y"])
START_VY: float = float(GAME["drone"]["start_vy"])

# --- speed -------------------------------------------------------------------------
# Per-level speeds live in levels.yaml (v1.7: L1 8 ... L5 16 m/s). The observation normalizes
# speed as (v - 8) / 8 (v1.7; it was (v - 6) / 8), so L1 = 0 and L5 = 1 as before.
SPEED_NORM_V0: float = 8.0
SURGE_MULT: float = float(GAME["speed"]["surge_mult"])
SURGE_FRAMES: int = int(round(float(GAME["speed"]["surge_s"]) * FPS))

# --- control ------------------------------------------------------------------------
_c = GAME["control"]
CTRL_GAIN: float = float(_c["gain"])
A_MAX: float = float(_c["a_max"])
V_HOLD: float = float(_c["v_hold"])
V_ASCEND: float = float(_c["v_ascend"])
V_DESCEND: float = float(_c["v_descend"])
JUMP_V: float = float(_c["jump_v"])
JUMP_BURST: int = int(_c["jump_burst_frames"])
JUMP_COOLDOWN: int = int(_c["jump_cooldown_frames"])
DUCK_BURST: int = int(_c["duck_burst_frames"])
DUCK_FRAMES: int = DUCK_BURST          # tucked frames = the dive burst (SPEC v1.7)
DUCK_V: float = float(_c["duck_v"])

# --- actions ------------------------------------------------------------------------
HOLD, ASCEND, DESCEND, JUMP, DUCK = 0, 1, 2, 3, 4
N_ACTIONS = 5
ACTION_NAMES = ("HOLD", "ASCEND", "DESCEND", "JUMP", "DUCK")

# --- hazards ------------------------------------------------------------------------
BUILDING, MOUNTAIN, BIRD, TRAFFIC, STORM, DOWNDRAFT, PLANE = 1, 2, 3, 4, 5, 6, 7
TREE, POLE, POWERLINE, KITE, CRANE = 8, 9, 10, 11, 12      # SPEC v2.0
HAZARD_NAMES = {
    BUILDING: "building",
    MOUNTAIN: "mountain",    # retired in v2.0 (weight 0 in every level)
    BIRD: "bird",
    TRAFFIC: "traffic",
    STORM: "storm",
    DOWNDRAFT: "downdraft",
    PLANE: "plane",          # SPEC v1.7; retired in v2.0
    TREE: "tree",
    POLE: "pole",            # street lights
    POWERLINE: "powerline",
    KITE: "kite",
    CRANE: "crane",
}
HAZARD_IDS = {v: k for k, v in HAZARD_NAMES.items()}
N_HAZARD_TYPES = 13          # type ids 0 (padding) .. 12
PART_KINDS = (TREE, POLE, POWERLINE, KITE, CRANE)          # statics built from boxes and wires (Static.parts)
SOLID_KINDS = (BUILDING, MOUNTAIN, BIRD, TRAFFIC, STORM, PLANE) + PART_KINDS   # can be crashed into / cleared
MOVING_KINDS = (BIRD, TRAFFIC, PLANE)
AIRBORNE_KINDS = (BIRD, TRAFFIC, PLANE, POWERLINE, CRANE)  # can be passed underneath
CRASH_CAUSES = ("ground", "ceiling", "building", "mountain", "bird", "traffic", "storm", "plane", "tree", "pole",
                "powerline", "kite", "crane")

HZ = GAME["hazards"]
DOWNDRAFT_ACCEL: float = float(HZ["downdraft"]["accel"])
STORM_RAMP_FRAC: float = float(HZ["storm"]["ramp_frac"])
PLANE_X_SLACK_S: float = float(HZ["plane"]["x_slack_s"])

# --- interruptions ------------------------------------------------------------------
GUST_S = tuple(float(v) for v in GAME["interruptions"]["gust_s"])
FOG_S = tuple(float(v) for v in GAME["interruptions"]["fog_s"])

# --- sensors ------------------------------------------------------------------------
_s = GAME["sensors"]
SENSOR_RANGE: float = float(_s["range_m"])
FOG_RANGE: float = float(_s["fog_range_m"])
FOG_NOISE: float = float(_s["fog_noise"])
N_RAYS: int = int(_s["n_rays"])
RAY_ANGLES_DEG = tuple(
    _s["ray_min_deg"] + i * (_s["ray_max_deg"] - _s["ray_min_deg"]) / (N_RAYS - 1) for i in range(N_RAYS)
)
RAY_COS = tuple(math.cos(math.radians(a)) for a in RAY_ANGLES_DEG)
RAY_SIN = tuple(0.0 if a == 0 else math.sin(math.radians(a)) for a in RAY_ANGLES_DEG)
MAX_HAZARDS: int = int(_s["max_hazards"])
NORM_RANGE: float = 16.0      # rays, dx and dist are divided by 16 (SPEC §5; v2.3 rows reach 20/16)
LEGACY_SENSOR_RANGE: float = 16.0   # sensor range for pilots made before v2.3
HAZ_COLS: int = 13            # v2.3 hazard row: 8 values + watched, seen low/high, predicted low/high
LEGACY_HAZ_COLS: int = 8
_tk = _s["tracker"]
TRACK_KEEP_S: float = float(_tk["keep_s"])
TRACK_MIN_FIT: int = int(_tk["min_fit_samples"])
TRACK_WATCH_NORM_S: float = float(_tk["watched_norm_s"])
TRACK_WINDOW_POINTS: int = int(_tk["window_points"])
TRACK_QUANTILES = tuple(float(q) for q in _tk["quantile_by_ahead"])


def hazard_cols(spec: tuple) -> int:
    """Values per hazard row for a sensor spec (v2.3: 13, before: 8)."""
    return HAZ_COLS if tuple(spec) >= (2, 3) else LEGACY_HAZ_COLS


def sensor_range(spec: tuple) -> float:
    """Clear-air sensor range for a sensor spec (v2.3: 20 m, before: 16 m)."""
    return SENSOR_RANGE if tuple(spec) >= (2, 3) else LEGACY_SENSOR_RANGE

# --- generator ----------------------------------------------------------------------
_g = GAME["generator"]
LOOKAHEAD: float = float(_g["lookahead_m"])
BATCH: float = float(_g["batch_m"])
CLONE_LOOKAHEAD: float = float(_g["clone_lookahead_m"])
START_CLEAR_S: float = float(_g["start_clear_s"])
MIN_GAP_S: float = float(_g["min_gap_s"])
CORRIDOR_MIN: float = float(_g["corridor_min_m"])
CLIMB_RATE: float = float(_g["climb_rate"])
DOWNDRAFT_CLIMB_RATE: float = float(_g["downdraft_climb_rate"])
CHECK_DX: float = float(_g["check_dx_m"])
MOVER_X_MARGIN: float = float(_g["mover_x_margin_m"])
EDGE_MARGIN: float = float(_g["edge_margin_m"])
MAX_TRIES: int = int(_g["max_tries"])
ROBUST: bool = bool(_g["robust"])
ROBUST_HALF: float = float(_g["robust_half_band_m"])
REACTION_FRAMES: int = int(_g["reaction_frames"])
ESCAPE_VY = tuple(float(v) for v in _g["escape_vy"])
ESCAPE_WAITS = tuple(int(v) for v in _g["escape_waits"])
PLANE_SPACING_S: float = float(_g["plane_spacing_s"])

# --- levels -------------------------------------------------------------------------
BASE_LEVELS = ("L1", "L2", "L3", "L4", "L5")
LEVELS = BASE_LEVELS + ("endless", "stress")
LEVEL_CODES = {name: i for i, name in enumerate(LEVELS)}   # int code, e.g. for datasets
ENDLESS_SEQUENCE = tuple(LEVELS_CFG["endless"]["sequence"])
ENDLESS_RAMP_M: float = float(LEVELS_CFG["endless"]["ramp_m"])
_l5p = LEVELS_CFG["endless"]["after_l5"]
L5P_GAP_SHRINK: float = float(_l5p["gap_shrink_per_km"])     # endless past L5 ("L5+", SPEC v1.7)
L5P_GAP_FLOOR_FRAC: float = float(_l5p["gap_floor_frac"])
L5P_MIN_GAP_S: float = float(_l5p["min_gap_s"])
L5P_COMBO_PER_KM: float = float(_l5p["combo_per_km"])
L5P_COMBO_MAX: float = float(_l5p["combo_max"])
