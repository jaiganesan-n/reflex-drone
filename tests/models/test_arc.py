"""The prototype's three checks, ported: hazard order, action order, padding."""
import numpy as np
import pytest
import torch

from models import FlatMLP, ArcNet, build_model, policy_outputs

N_HAZ = 12


def sample(B=2, n_valid=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    haz_mask = torch.zeros(B, N_HAZ, dtype=torch.bool)
    haz_mask[:, :n_valid] = True
    avail = torch.ones(B, 5, dtype=torch.bool)
    avail[:, 3] = False                                   # JUMP cooling down
    return dict(self_f=torch.randn(B, 11, generator=g), rays=torch.rand(B, 13, generator=g),
                haz=torch.randn(B, N_HAZ, 8, generator=g), haz_type=torch.randint(0, 7, (B, N_HAZ), generator=g),
                haz_mask=haz_mask, act_ids=torch.arange(5)[None].repeat(B, 1),
                act_desc=torch.randn(B, 5, 3, generator=g), act_avail=avail)


@pytest.fixture(scope="module", params=[dict(), dict(d=48, heads=4, enc_layers=1, head_layers=1),
                                        dict(ttc_features=True)],
                ids=["arc", "arc-small", "arc-ttc"])
def net(request):
    torch.manual_seed(0)
    return ArcNet(**request.param).eval()


@torch.no_grad()
def test_hazard_order_gives_identical_outputs(net):
    x = sample()
    out = net(**x)
    perm = torch.cat([torch.randperm(7, generator=torch.Generator().manual_seed(3)), torch.arange(7, N_HAZ)])
    y = dict(x, haz=x["haz"][:, perm], haz_type=x["haz_type"][:, perm], haz_mask=x["haz_mask"][:, perm])
    out2 = net(**y)
    for a, b in zip(out[:3], out2[:3]):
        assert torch.allclose(a, b, atol=1e-5)
    # decide_attn follows the hazards it attends to
    assert torch.allclose(out2[3], out[3][:, perm], atol=1e-6)
    assert torch.allclose(out[4], out2[4], atol=1e-5)


@torch.no_grad()
def test_action_order_permutes_logits(net):
    x = sample()
    lg, sf, th, _, _ = net(**x)
    ap = torch.tensor([4, 2, 0, 3, 1])
    z = dict(x, act_ids=x["act_ids"][:, ap], act_desc=x["act_desc"][:, ap], act_avail=x["act_avail"][:, ap])
    lg2, sf2, th2, _, _ = net(**z)
    assert torch.allclose(lg2, lg[:, ap], atol=1e-5)
    assert torch.allclose(sf2, sf[:, ap], atol=1e-5)
    assert torch.allclose(th2, th, atol=1e-5)


@torch.no_grad()
def test_padding_junk_changes_nothing(net):
    x = sample()
    out = net(**x)
    w = dict(x, haz=x["haz"].clone(), haz_type=x["haz_type"].clone())
    w["haz"][:, 7:] = 99.0
    w["haz_type"][:, 7:] = 5
    out2 = net(**w)
    for a, b in zip(out, out2):
        assert torch.allclose(a, b, atol=1e-5)
    assert torch.all(out[3][:, 7:] == 0)                 # no attention on padding
    assert torch.allclose(out[3].sum(-1) <= 1.0 + 1e-6, torch.ones(2, dtype=torch.bool))


@torch.no_grad()
def test_masked_actions_are_minus_inf_and_no_hazards_is_fine(net):
    x = sample(n_valid=0)
    lg, sf, th, attn, v = net(**x)
    assert torch.isinf(lg[:, 3]).all() and torch.isfinite(lg[:, [0, 1, 2, 4]]).all()
    assert torch.isfinite(sf).all() and torch.isfinite(th).all() and torch.all(attn == 0)


def test_mlp_shapes_and_policy_outputs_mask():
    m = build_model("mlp")
    assert isinstance(m, FlatMLP)
    assert sum(p.numel() for p in m.body.parameters()) + sum(p.numel() for p in m.pi.parameters()) == \
        80 * 128 + 128 + 128 * 64 + 64 + 64 * 5 + 5
    obs = torch.randn(3, 80)
    avail = torch.tensor([[1, 1, 1, 0, 1]] * 3, dtype=torch.bool)
    out = policy_outputs(m, "mlp", {"obs": obs, "act_avail": avail})
    assert out["logits"].shape == (3, 5) and torch.isinf(out["logits"][:, 3]).all()


def test_ttc_features_values_and_parameter_count():
    """Drone at 5 m; a building 0-6 m tall, 8 m ahead, closing at 8 m/s: 1 s to contact, climb 1.175 m
    (bottom of the hitbox over the roof), descent 5.175 m, ASCEND covers 3 * 1 - 0.375 = 2.625 m."""
    from models.arc import hazard_physics
    self_f = torch.zeros(1, 11)
    self_f[0, 0] = 0.5
    haz = torch.zeros(1, 12, 8)
    haz[0, 0, :5] = torch.tensor([8 / 16, 0.0, 0.6, 0.25, -8 / 16])
    f = hazard_physics(self_f, haz)[0, 0]
    want = torch.tensor([1 / 3, 0.1175, 0.5175, (2.625 - 1.175) / 5, -1.0 * min(1.0, (5.175 - 2.625) / 5)])
    assert torch.allclose(f, want, atol=1e-5)
    assert sum(p.numel() for p in ArcNet().parameters()) == 129_351 + 5 * 64          # v2.0: 13 hazard types (was 8)
    assert sum(p.numel() for p in ArcNet(ttc_features=True).parameters()) == 129_351 + 10 * 64
    assert build_model("arc", {"ttc_features": True}).ttc_features
