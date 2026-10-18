"""Contract loss (SPEC §13): masked actions are excluded and get no gradient."""
import numpy as np
import torch

from teacher.planner import soft_targets
from train.common import auc, contract_loss, top_set_agreement


def _batch(B=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    avail = torch.ones(B, 5, dtype=torch.bool)
    avail[:, 3] = False
    avail[0, 1:] = False                                # jump burst: only HOLD
    scores = torch.randn(B, 5, generator=g).numpy() * 3 + 30
    scores[~avail.numpy()] = -np.inf
    return {"act_avail": avail, "p_teacher": torch.from_numpy(soft_targets(scores, 2.0)),
            "safe": (torch.rand(B, 5, generator=g) > 0.5).float(), "threat": torch.randint(0, 5, (B,), generator=g)}


def test_masked_actions_get_no_gradient():
    b = _batch()
    logits = torch.randn(6, 5, requires_grad=True)
    safe = torch.randn(6, 5, requires_grad=True)
    threat = torch.randn(6, 5, requires_grad=True)
    loss, parts = contract_loss({"logits": logits, "safe_logits": safe, "threat_logits": threat}, b, "arc")
    loss.backward()
    m = ~b["act_avail"]
    assert torch.isfinite(loss)
    assert torch.all(logits.grad[m] == 0) and torch.all(safe.grad[m] == 0)
    assert logits.grad[~m].abs().sum() > 0 and safe.grad[~m].abs().sum() > 0
    assert set(parts) == {"kl", "bce", "ce", "loss"}


def test_already_masked_logits_and_mlp_kl_only():
    b = _batch()
    raw = torch.randn(6, 5, requires_grad=True)
    masked = raw.masked_fill(~b["act_avail"], float("-inf"))
    loss, parts = contract_loss({"logits": masked}, b, "mlp")
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"kl", "loss"}
    assert torch.all(raw.grad[~b["act_avail"]] == 0)


def test_kl_is_zero_at_the_teacher_distribution():
    b = _batch()
    logits = torch.log(b["p_teacher"].clamp_min(1e-30))
    _, parts = contract_loss({"logits": logits}, b, "mlp")
    assert parts["kl"].item() < 1e-6


def test_top_set_agreement_counts_ties():
    pt = torch.tensor([[0.3, 0.3, 0.3, 0.0, 0.1], [0.7, 0.1, 0.1, 0.0, 0.1]])
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.0]])
    assert top_set_agreement(logits, pt).tolist() == [True, False]


def test_auc():
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    assert np.isnan(auc(np.array([0.1]), np.array([1])))
