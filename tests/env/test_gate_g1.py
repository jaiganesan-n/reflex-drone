"""Gate G1 (SPEC §16): the rule agent averages >= 300 m on L1 VAL seeds."""
import numpy as np

from app.rule_eval import evaluate


def test_rule_agent_averages_300m_on_L1_val():
    res = evaluate("rule", "L1", cap=3000)
    assert len(res) == 50
    mean = float(np.mean([r["distance_m"] for r in res]))
    assert mean >= 300.0, f"rule agent mean distance {mean:.1f} m"
