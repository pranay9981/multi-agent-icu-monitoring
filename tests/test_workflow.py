"""Tests for XGBoostInference, RuntimePreprocessor, and AgenticICUWorkflow edge paths."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_icu.domain.contracts import (
    EvaluatePatientRequest,
    ModelAgentResult,
    ObservationRecord,
)
from agentic_icu.domain.features import DYNAMIC_FEATURES, STATIC_FEATURES
from agentic_icu.inference.tabular import XGBoostInference
from agentic_icu.orchestration.workflow import AgenticICUWorkflow
from agentic_icu.preprocessing.windowing import RuntimePreprocessor

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_statistics(tmp_path: Path) -> Path:
    """Write a minimal train_statistics.json that covers every DYNAMIC_FEATURE."""
    stats = {
        "fill_medians": {f: 0.0 for f in DYNAMIC_FEATURES},
        "value_means": {f: 0.0 for f in DYNAMIC_FEATURES},
        "value_stds": {f: 1.0 for f in DYNAMIC_FEATURES},
        "static_fill_values": {f: 0.0 for f in STATIC_FEATURES},
    }
    p = tmp_path / "train_statistics.json"
    p.write_text(json.dumps(stats))
    return p


def _make_pipeline_config(tmp_path: Path, obs_hours: int = 24, seq_hours: int = 72) -> Path:
    cfg = {"observation_hours": obs_hours, "patient_seq_max_hours": seq_hours}
    p = tmp_path / "pipeline_config.json"
    p.write_text(json.dumps(cfg))
    return p


def _make_preprocessor(tmp_path: Path) -> RuntimePreprocessor:
    stats_p = _make_statistics(tmp_path)
    cfg_p = _make_pipeline_config(tmp_path)
    rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
    rp.load()
    return rp


def _records(n: int = 6) -> list[ObservationRecord]:
    return [
        ObservationRecord(values={"HR": float(70 + i), "ICULOS": float(i + 1)})
        for i in range(n)
    ]


# ── RuntimePreprocessor ───────────────────────────────────────────────────────


class TestAgenticICUWorkflow:
    def _make_workflow(
        self,
        resp_agent=None,
        ensemble=None,
        sq_valid=True,
        sq_suppress=False,
    ):
        from agentic_icu.agents.reasoner import AlertPolicy, ClinicalReasoner
        from agentic_icu.domain.contracts import SignalQualityResult

        policy = AlertPolicy()
        reasoner = ClinicalReasoner(policy)

        # Use a real SignalQualityResult so Pydantic validation passes
        real_sq = SignalQualityResult(
            signal_valid=sq_valid,
            suppression_recommendation=sq_suppress,
            suppression_mode="full" if sq_suppress else "none",
        )

        sq_agent = MagicMock()
        sq_agent.evaluate.return_value = (real_sq, [])

        vitals_agent = MagicMock()
        vitals_agent.evaluate.return_value = (
            ModelAgentResult(
                status="available", score=0.8, risk_band="high",
                detail="high", decision_threshold=0.5, threshold_ratio=1.6,
            ),
            [],
        )

        lab_agent = MagicMock()
        lab_agent.evaluate.return_value = (
            ModelAgentResult(
                status="available", score=0.3, risk_band="low",
                detail="low", decision_threshold=0.5, threshold_ratio=0.6,
            ),
            [],
        )

        return AgenticICUWorkflow(
            signal_quality_agent=sq_agent,
            vitals_agent=vitals_agent,
            lab_agent=lab_agent,
            reasoner=reasoner,
            resp_failure_agent=resp_agent,
            ensemble=ensemble,
        )

    def _request(self) -> EvaluatePatientRequest:
        return EvaluatePatientRequest(
            patient_id="test",
            observation_window=_records(6),
        )

    def test_full_suppression_short_circuits(self):
        wf = self._make_workflow(sq_valid=False, sq_suppress=True)
        response = wf.evaluate(self._request())
        assert response.vitals_agent.status == "unavailable"
        assert response.lab_agent.status == "unavailable"
        # Vitals/Lab agents should NOT be called at all
        wf.vitals_agent.evaluate.assert_not_called()
        wf.lab_agent.evaluate.assert_not_called()

    def test_no_resp_agent_returns_unavailable(self):
        wf = self._make_workflow(resp_agent=None)
        response = wf.evaluate(self._request())
        assert response.resp_failure_agent.status == "unavailable"
        assert "not configured" in response.resp_failure_agent.detail

    def test_with_resp_agent_calls_it(self):
        resp_mock = MagicMock()
        resp_mock.evaluate.return_value = (
            ModelAgentResult(status="available", score=0.2, risk_band="low", detail="ok"),
            [],
        )
        wf = self._make_workflow(resp_agent=resp_mock)
        wf.evaluate(self._request())
        resp_mock.evaluate.assert_called_once()

    def test_ensemble_predict_error_is_tolerated(self):
        from agentic_icu.inference.ensemble import EnsembleInference

        mock_ensemble = MagicMock(spec=EnsembleInference)
        mock_ensemble.available = True
        mock_ensemble.predict.side_effect = ValueError("bad scores")

        wf = self._make_workflow(ensemble=mock_ensemble)
        # Should not raise — error is caught internally
        response = wf.evaluate(self._request())
        assert response.ensemble_agent.status == "unavailable"

    def test_ensemble_not_called_when_unavailable(self):
        from agentic_icu.inference.ensemble import EnsembleInference

        mock_ensemble = MagicMock(spec=EnsembleInference)
        mock_ensemble.available = False

        wf = self._make_workflow(ensemble=mock_ensemble)
        wf.evaluate(self._request())
        mock_ensemble.predict.assert_not_called()

    def test_ensemble_called_when_available(self):
        from agentic_icu.inference.ensemble import EnsembleInference

        mock_ensemble = MagicMock(spec=EnsembleInference)
        mock_ensemble.available = True
        mock_ensemble.predict.return_value = 0.88

        wf = self._make_workflow(ensemble=mock_ensemble)
        response = wf.evaluate(self._request())
        mock_ensemble.predict.assert_called_once()
        assert response.ensemble_agent.status == "available"

    def test_ensemble_high_risk_band(self):
        from agentic_icu.inference.ensemble import EnsembleInference
        mock_ensemble = MagicMock(spec=EnsembleInference)
        mock_ensemble.available = True
        mock_ensemble.predict.return_value = 0.9  # above high_thresh=0.8

        wf = self._make_workflow(ensemble=mock_ensemble)
        response = wf.evaluate(self._request())
        assert response.ensemble_agent.risk_band == "high"

    def test_ensemble_moderate_risk_band(self):
        from agentic_icu.inference.ensemble import EnsembleInference
        mock_ensemble = MagicMock(spec=EnsembleInference)
        mock_ensemble.available = True
        mock_ensemble.predict.return_value = 0.65  # between 0.55 and 0.8

        wf = self._make_workflow(ensemble=mock_ensemble)
        response = wf.evaluate(self._request())
        assert response.ensemble_agent.risk_band == "moderate"

    def test_response_contains_all_required_fields(self):
        wf = self._make_workflow()
        response = wf.evaluate(self._request())
        assert response.patient_id == "test"
        assert response.clinical_decision is not None
        assert response.reasoning_log is not None
