from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestReasonerUncoveredPaths(unittest.TestCase):
    def test_alert_policy_rejects_threshold_above_one(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        with self.assertRaises(ValueError):
            AlertPolicy(high_alert_extreme_sequence_score_threshold=1.5)

    def test_alert_policy_rejects_negative_threshold(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        with self.assertRaises(ValueError):
            AlertPolicy(medium_alert_sequence_score_threshold=-0.1)

    def test_alert_policy_accepts_valid_probability_thresholds(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        policy = AlertPolicy(
            high_alert_extreme_sequence_score_threshold=0.9,
            medium_alert_sequence_score_threshold=0.5,
        )
        self.assertEqual(policy.high_alert_extreme_sequence_score_threshold, 0.9)

    def test_alert_policy_accepts_none_thresholds(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        policy = AlertPolicy(high_alert_max_score_threshold=None)
        self.assertIsNone(policy.high_alert_max_score_threshold)

    def test_alert_policy_rejects_suppression_factor_above_one(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        with self.assertRaises(ValueError):
            AlertPolicy(partial_suppression_factor=1.5)

    def test_alert_policy_rejects_suppression_factor_below_zero(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        with self.assertRaises(ValueError):
            AlertPolicy(partial_suppression_factor=-0.1)

    def test_alert_policy_accepts_valid_suppression_factor(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        policy = AlertPolicy(partial_suppression_factor=0.5)
        self.assertEqual(policy.partial_suppression_factor, 0.5)


if __name__ == "__main__":
    unittest.main()
