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


class TestRuntimePreprocessor:
    def test_available_false_when_files_missing(self, tmp_path):
        rp = RuntimePreprocessor(
            str(tmp_path / "stats.json"), str(tmp_path / "cfg.json")
        )
        assert rp.available is False

    def test_available_true_when_files_exist(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        assert rp.available is True

    def test_load_raises_when_fill_medians_missing(self, tmp_path):
        stats = {
            "value_means": {f: 0.0 for f in DYNAMIC_FEATURES},
            "value_stds": {f: 1.0 for f in DYNAMIC_FEATURES},
            "static_fill_values": {},
        }
        stats_p = tmp_path / "stats.json"
        stats_p.write_text(json.dumps(stats))
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        with pytest.raises(ValueError, match="fill_medians missing"):
            rp.load()

    def test_load_raises_when_value_means_missing(self, tmp_path):
        stats = {
            "fill_medians": {f: 0.0 for f in DYNAMIC_FEATURES},
            "value_stds": {f: 1.0 for f in DYNAMIC_FEATURES},
            "static_fill_values": {},
        }
        stats_p = tmp_path / "stats.json"
        stats_p.write_text(json.dumps(stats))
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        with pytest.raises(ValueError, match="value_means missing"):
            rp.load()

    def test_load_raises_when_value_stds_missing(self, tmp_path):
        stats = {
            "fill_medians": {f: 0.0 for f in DYNAMIC_FEATURES},
            "value_means": {f: 0.0 for f in DYNAMIC_FEATURES},
            "static_fill_values": {},
        }
        stats_p = tmp_path / "stats.json"
        stats_p.write_text(json.dumps(stats))
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        with pytest.raises(ValueError, match="value_stds missing"):
            rp.load()

    def test_load_raises_when_static_fill_values_missing(self, tmp_path):
        stats = {
            "fill_medians": {f: 0.0 for f in DYNAMIC_FEATURES},
            "value_means": {f: 0.0 for f in DYNAMIC_FEATURES},
            "value_stds": {f: 1.0 for f in DYNAMIC_FEATURES},
        }
        stats_p = tmp_path / "stats.json"
        stats_p.write_text(json.dumps(stats))
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        with pytest.raises(ValueError, match="static_fill_values"):
            rp.load()

    def test_load_is_idempotent(self, tmp_path):
        rp = _make_preprocessor(tmp_path)
        first_stats = id(rp._stats)
        rp.load()  # second call — should be a no-op
        assert id(rp._stats) == first_stats

    def test_stats_property_triggers_lazy_load(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        assert rp._stats is None
        _ = rp.stats
        assert rp._stats is not None

    def test_pipeline_config_property_triggers_lazy_load(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        _ = rp.pipeline_config
        assert rp._pipeline_config is not None

    def test_observation_hours_defaults_to_24(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = tmp_path / "cfg.json"
        cfg_p.write_text(json.dumps({}))  # empty — should default
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        assert rp.observation_hours == 24

    def test_sequence_hours_defaults_to_observation_hours(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = tmp_path / "cfg.json"
        cfg_p.write_text(json.dumps({"observation_hours": 24}))
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        assert rp.sequence_hours == 24

    def test_build_tabular_features_returns_dict(self, tmp_path):
        rp = _make_preprocessor(tmp_path)
        features = rp.build_tabular_features(_records(10))
        assert isinstance(features, dict)
        assert "HR__last" in features
        assert "window_missing_fraction" in features

    def test_build_tabular_features_truncates_long_window(self, tmp_path):
        # Use obs_hours=4 so a 30-row window gets truncated
        stats_p = _make_statistics(tmp_path)
        cfg_p = _make_pipeline_config(tmp_path, obs_hours=4, seq_hours=72)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        rp.load()
        features = rp.build_tabular_features(_records(30))
        assert isinstance(features, dict)

    def test_build_sequence_tensor_shape(self, tmp_path):
        rp = _make_preprocessor(tmp_path)
        tensor = rp.build_sequence_tensor(_records(6))
        assert tensor.ndim == 2
        assert tensor.shape[1] == len(DYNAMIC_FEATURES) * 2  # values + masks

    def test_build_sequence_tensor_pads_short_window(self, tmp_path):
        # seq_hours=72 but only 6 records — should pad
        rp = _make_preprocessor(tmp_path)
        tensor = rp.build_sequence_tensor(_records(6))
        assert tensor.shape[0] == 72  # padded to seq_hours

    def test_build_sequence_tensor_truncates_long_window(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = _make_pipeline_config(tmp_path, obs_hours=24, seq_hours=10)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        rp.load()
        tensor = rp.build_sequence_tensor(_records(30))
        assert tensor.shape[0] == 10

    def test_records_to_frame_fills_iculos_from_index(self, tmp_path):
        rp = _make_preprocessor(tmp_path)
        records = [ObservationRecord(values={"HR": 80.0})]  # no ICULOS
        df = rp.records_to_frame(records)
        assert df["ICULOS"].iloc[0] == 1.0  # index 1

    def test_slope_6h_returns_zero_for_single_value(self):
        slope = RuntimePreprocessor._slope_6h(np.array([5.0]))
        assert slope == 0.0

    def test_composite_features_populated(self, tmp_path):
        rp = _make_preprocessor(tmp_path)
        features = rp.build_tabular_features(_records(10))
        assert "shock_index" in features
        assert "pulse_pressure" in features
        assert "qsofa_score" in features
        assert "HR__slope_6h" in features


# ── XGBoostInference ──────────────────────────────────────────────────────────


