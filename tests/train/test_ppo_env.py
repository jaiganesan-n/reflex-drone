"""PPO training envs (train/ppo.py): the game reward by default, the optional training-only
shaping (clearance bonus, change and burst costs) and the per-episode level mix."""
import collections

import numpy as np
import pytest

from drone_env import constants as C
from drone_env.env import DroneDashEnv
from train.ppo import CurriculumEnv

ACTS = [0, 0, 1, 1, 0, C.JUMP, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0]


def pair(shaping=None, cap=3000):
    plain, other = CurriculumEnv(3, 0, "val", cap, "L1"), CurriculumEnv(3, 0, "val", cap, "L1", shaping=shaping)
    plain.reset()
    other.reset()
    return plain, other


def test_no_shaping_is_the_game_reward():
    env, _ = pair()
    game = DroneDashEnv("L1", max_frames=3000)
    game.reset(seed=env.env.sim_seed)
    for a in ACTS:
        assert env.step(a)[1] == game.step(a)[1]


def test_change_and_burst_costs_are_exact():
    plain, env = pair({"change_penalty": 0.01, "burst_penalty": 0.1})
    diffs = []
    for a in ACTS:
        diffs.append(env.step(a)[1] - plain.step(a)[1])
    executed_changes = [i for i in range(len(ACTS)) if i in (2, 4, 5, 6, 15, 17)]   # JUMP's burst masks to HOLD
    for i, d in enumerate(diffs):
        want = -0.01 * (i in executed_changes) - 0.1 * (i == 5)
        assert d == pytest.approx(want), i


def test_clearance_bonus_is_full_in_open_sky_and_small_under_the_ceiling():
    plain, env = pair({"clearance_bonus": 0.02, "clearance_cap_m": 2.0})
    d0 = env.step(0)[1] - plain.step(0)[1]
    assert env.env.sim.clearance() >= 2.0 and d0 == pytest.approx(0.02)
    for e in (plain, env):
        e.env.sim.y, e.env.sim.vy = 9.5, 0.0
    d1 = env.step(0)[1] - plain.step(0)[1]
    cl = env.env.sim.clearance()
    assert cl < 0.5 and d1 == pytest.approx(0.02 * cl / 2.0) and d1 < 0.005


def test_episode_stats_and_level_mix():
    env = CurriculumEnv(1, 0, "train", 20, "L1", level_mix={"L2": 1.0, "L4": 3.0},
                        shaping={"clearance_bonus": 0.01})
    env.reset()
    first = env.level
    infos = [env.step(a)[-1] for a in ACTS + [0, 0]]
    done = [i for i in infos if i["ep_done"]]
    assert len(done) == 1 and done[0]["ep_frames"] == 20 and done[0]["ep_level"] == C.LEVEL_CODES[first]
    assert done[0]["ep_changes"] == 6 and done[0]["ep_bursts"] == 1 and done[0]["ep_clearance"] > 1.0
    levels = collections.Counter()
    for _ in range(200):
        env._new_episode()
        levels[env.level] += 1
    assert set(levels) == {"L2", "L4"} and 0.65 < levels["L4"] / 200 < 0.85
    assert np.isclose(env.mix[1].sum(), 1.0)


def _scene(env, plane_ahead):
    """Replace the env's world by an empty calm L5 sky, drone at 5 m, optionally a plane 4 m ahead at its height."""
    from drone_env import events
    from tests.env.helpers import empty_sim
    sim = empty_sim(5.0, "L5")
    events.make_calm(sim)
    if plane_ahead:
        sim.place_mover(C.PLANE, sim.x + C.HALF_W + 4.0, 5.0, 5.0, 1.0, s=8.0)   # 0.2 s at 12 + 8 m/s
    env.env.sim = sim


@pytest.mark.parametrize("plane_ahead, needless", [(False, True), (True, False)])
def test_needless_burst_costs_only_when_hold_ascend_or_descend_was_safe(plane_ahead, needless):
    """Open sky: a JUMP is needless. A plane 4 m ahead at the drone's height at L5: only JUMP/DUCK
    are safe (teacher check), so the JUMP pays only the ordinary burst cost."""
    base = CurriculumEnv(0, 0, "val", 3000, "L5", shaping={"burst_penalty": 0.1})
    env = CurriculumEnv(0, 0, "val", 3000, "L5", shaping={"burst_penalty": 0.1, "needless_burst_penalty": 1.0})
    for e in (base, env):
        e.reset()
        _scene(e, plane_ahead)
    d = env.step(C.JUMP)[1] - base.step(C.JUMP)[1]
    assert d == pytest.approx(-1.0 if needless else 0.0)
    assert env.ep_needless == int(needless) and env.ep_bursts == 1


def test_target_change_and_vertical_speed_costs():
    """ASCEND -> HOLD -> DESCEND costs what ASCEND -> DESCEND does; climbing costs per |vy|."""
    def cost(actions, **sh):
        plain, env = pair(sh)
        return sum(env.step(a)[1] - plain.step(a)[1] for a in actions)
    via_hold = cost([1, 1, 0, 2, 2], target_change_penalty=0.01)
    direct = cost([1, 1, 2, 2, 2], target_change_penalty=0.01)
    assert via_hold == pytest.approx(direct) == pytest.approx(-0.01 * 3.0 * 3)   # 0 -> 3 -> 0 -> -3 vs 0 -> 3 -> -3
    hold, climb = cost([0] * 10, vertical_speed_penalty=0.003), cost([1] * 10, vertical_speed_penalty=0.003)
    assert hold == pytest.approx(0.0, abs=1e-6) and climb < -0.003 * 5


def test_mask_needless_bursts_in_teacher_targets():
    import torch
    from train.ppo import mask_needless_bursts
    p = torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1], [0.1, 0.1, 0.1, 0.4, 0.3]])
    safe = torch.tensor([[1.0, 0, 0, 1, 1], [0.0, 0, 0, 1, 0]])        # row 0: HOLD safe; row 1: only JUMP is
    q = mask_needless_bursts(p, safe)
    assert torch.allclose(q[0], torch.tensor([0.5, 0.25, 0.25, 0.0, 0.0])) and torch.allclose(q[1], p[1])


@pytest.mark.parametrize("level, factor", [("L1", 0.75), ("L3", 0.5), ("L5", 0.5)])   # 10 / 12 / 12 m/s (v2.3)
def test_slow_burst_penalty_fades_from_8_to_16_mps(level, factor):
    from drone_env import events
    from tests.env.helpers import empty_sim
    base = CurriculumEnv(0, 0, "val", 3000, level, shaping={"burst_penalty": 0.1})
    env = CurriculumEnv(0, 0, "val", 3000, level, shaping={"burst_penalty": 0.1, "slow_burst_penalty": 1.0})
    for e in (base, env):
        e.reset()
        sim = empty_sim(5.0, level)
        events.make_calm(sim)
        e.env.sim = sim
    d = env.step(C.JUMP)[1] - base.step(C.JUMP)[1]
    assert d == pytest.approx(-factor)


def test_needless_check_reads_the_teacher_settings_file(tmp_path):
    f = tmp_path / "teacher.yaml"
    f.write_text("sensor_limited: true\n")
    env = CurriculumEnv(0, 0, "val", 3000, "L5", shaping={"needless_burst_penalty": 1.0, "needless_safe_frames": 45},
                        teacher_config=str(f))
    env.reset()
    _scene(env, False)
    assert env._needless_burst(C.JUMP)
    assert env.teacher_cfg.sensor_limited and env.teacher_cfg.safe_frames == 45


def test_hard_label_term_is_off_by_default_and_adds_ce_on_the_teacher_action():
    import torch
    from train.common import contract_loss
    logits = torch.tensor([[1.0, 0.5, 0.2, 0.0, -1.0]])
    batch = {"act_avail": torch.tensor([[True, True, True, False, True]]),
             "p_teacher": torch.tensor([[0.4, 0.3, 0.3, 0.0, 0.0]]), "action": torch.tensor([1])}
    out = {"logits": logits}
    base, parts = contract_loss(out, batch, "mlp")
    hard, parts_h = contract_loss(out, batch, "mlp", {"hard": 0.5})
    ce = -torch.log_softmax(logits.masked_fill(~batch["act_avail"], float("-inf")), -1)[0, 1]
    assert "hard" not in parts and torch.isclose(parts_h["hard"], ce)
    assert torch.isclose(hard, base + 0.5 * ce)


def test_path_bonus_pays_for_a_clear_level_path_ahead():
    """Open sky: the full bonus every frame. A building in the drone's band 8 m ahead at L1 (1.0 s,
    horizon 1.2 s): the bonus is 1.0 / 1.2 of it, minus the frame the drone flies closer."""
    plain, env = pair({"path_bonus": 0.01, "path_horizon_s": 1.2})
    d0 = env.step(0)[1] - plain.step(0)[1]
    assert d0 == pytest.approx(0.01)
    for e in (plain, env):
        s = e.env.sim
        s.statics, s.pending, s.movers = [], [], []
        nose = s.distance_m + C.DRONE_X + C.HALF_W + s.speed * C.DT
        s.place_static(C.BUILDING, nose + 8.0, nose + 10.0, 0.0, C.CEILING - 0.5)
    d1 = env.step(0)[1] - plain.step(0)[1]
    assert d1 == pytest.approx(0.01 * env.env.sim.path_time(1.2, 0.3) / 1.2) and 0.0 < d1 < 0.009


def test_anchor_kl_to_the_starting_policy():
    """ref_kl: 0 for the starting policy itself, > 0 once it moves, masked actions add nothing and
    get a finite (zero) gradient."""
    import torch
    from train.ppo import ref_kl
    avail = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    start = torch.tensor([[2.0, 0.0, -1.0, 0.0, 0.0], [0.5, 0.1, -0.3, -2.0, 1.0]]).masked_fill(~avail, float("-inf"))
    ref_logp = torch.log_softmax(start / 0.32, -1)
    assert float(ref_kl(start, ref_logp, avail, 0.32)) == pytest.approx(0.0, abs=1e-6)
    moved = start.clone().requires_grad_(True)
    with torch.no_grad():
        moved[:, 1] += 1.0
    kl = ref_kl(moved, ref_logp, avail, 0.32)
    kl.backward()
    p = torch.softmax(moved.detach() / 0.32, -1)
    want = (p * (torch.log_softmax(moved.detach() / 0.32, -1) - ref_logp)).nan_to_num(0.0).sum(-1).mean()
    assert float(kl) == pytest.approx(float(want)) and float(kl) > 0.1
    assert torch.isfinite(moved.grad).all() and (moved.grad[0, 3:] == 0).all()
