from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestClinicalReasoner(unittest.TestCase):
    def setUp(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy, ClinicalReasoner
        from agentic_icu.domain.contracts import ModelAgentResult, SignalQualityResult

        self.policy = AlertPolicy()
        self.reasoner = ClinicalReasoner(self.policy)
        self.ModelAgentResult = ModelAgentResult
        self.SignalQualityResult = SignalQualityResult

    def _sq_clear(self):
        return self.SignalQualityResult(signal_valid=True, suppression_recommendation=False, suppression_mode="none")

    def _sq_full_suppress(self):
        return self.SignalQualityResult(signal_valid=False, suppression_recommendation=True, suppression_mode="full")

    def _sq_partial_suppress(self):
        return self.SignalQualityResult(
            signal_valid=True, suppression_recommendation=True, suppression_mode="partial",
            artifact_type="soft_range_violation", artifact_affected_features=["Temp"],
        )

    def _agent(self, score: float) -> "ModelAgentResult":
        dt = 0.5
        tr = score / dt
        return self.ModelAgentResult(
            status="available", score=score, risk_band="high" if tr >= 1 else "low",
            detail=f"score {score:.3f}", decision_threshold=dt, threshold_ratio=tr,
        )

    def test_full_suppression_blocks_alert(self) -> None:
        decision, *_ = self.reasoner.decide(
            self._sq_full_suppress(), self._agent(0.99), self._agent(0.99)
        )
        self.assertFalse(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.suppressed_artifact_alert_type)

    def test_high_alert_triggers_on_extreme_sequence(self) -> None:
        decision, *_ = self.reasoner.decide(
            self._sq_clear(), self._agent(0.95), self._agent(0.1)
        )
        self.assertTrue(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.high_alert_type)

    def test_stable_when_below_thresholds(self) -> None:
        decision, *_ = self.reasoner.decide(
            self._sq_clear(), self._agent(0.1), self._agent(0.05)
        )
        self.assertFalse(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.stable_alert_type)

    def test_medium_alert_on_sequence_threshold(self) -> None:
        decision, *_ = self.reasoner.decide(
            self._sq_clear(), self._agent(0.6), self._agent(0.05)
        )
        self.assertTrue(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.medium_alert_type)

    def test_partial_suppression_reduces_scores_consistently(self) -> None:
        high_vitals = self._agent(0.95)
        decision, logs, adj_vitals, adj_lab, _ = self.reasoner.decide(
            self._sq_partial_suppress(), high_vitals, self._agent(0.1)
        )
        suppressed_score = 0.95 * self.policy.partial_suppression_factor
        self.assertAlmostEqual(adj_vitals.score, suppressed_score, places=5)
        self.assertIn("suppression-adjusted", adj_vitals.detail)
        expected_ratio = suppressed_score / high_vitals.decision_threshold
        self.assertAlmostEqual(adj_vitals.threshold_ratio, expected_ratio, places=5)
        suppression_log = next((l for l in logs if "suppression" in l.message.lower()), None)
        self.assertIsNotNone(suppression_log)
        self.assertTrue(decision.alert_triggered)

    def test_resp_high_alert_when_sepsis_stable(self) -> None:
        from agentic_icu.domain.contracts import ModelAgentResult
        resp_result = ModelAgentResult(
            status="available", score=0.9, risk_band="high",
            detail="resp score 0.9", decision_threshold=0.4, threshold_ratio=2.25,
        )
        decision, *_ = self.reasoner.decide(
            self._sq_clear(), self._agent(0.1), self._agent(0.05), resp_result
        )
        self.assertTrue(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.resp_high_alert_type)

    def test_no_available_scores_returns_models_unavailable(self) -> None:
        from agentic_icu.domain.contracts import ModelAgentResult
        unavailable = ModelAgentResult(status="unavailable", detail="no model")
        decision, *_ = self.reasoner.decide(self._sq_clear(), unavailable, unavailable)
        self.assertFalse(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.models_unavailable_alert_type)

    def test_resp_high_alert_when_sepsis_models_unavailable(self) -> None:
        from agentic_icu.domain.contracts import ModelAgentResult
        unavailable = ModelAgentResult(status="unavailable", detail="no model")
        resp_result = ModelAgentResult(
            status="available", score=0.9, risk_band="high",
            detail="resp score 0.9", decision_threshold=0.4, threshold_ratio=2.25,
        )
        decision, *_ = self.reasoner.decide(self._sq_clear(), unavailable, unavailable, resp_result)
        self.assertTrue(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.resp_high_alert_type)

if __name__ == "__main__":
    unittest.main()
