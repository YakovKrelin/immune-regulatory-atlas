from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from hypothesis_engine_v0_1.scoring_rules import ScoringPolicy


def _load_csv_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "csv_to_observations.py"
    spec = importlib.util.spec_from_file_location("csv_to_observations_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_csv_with_bom_preserves_order_and_boundary_confidences(tmp_path: Path):
    csv_mod = _load_csv_module()
    csv_path = tmp_path / "weighted_bom.csv"
    body = (
        "observation_id,direction,confidence,source_type\n"
        "obs-high,support,1.0,assay\n"
        "obs-zero,contradict,0.0,literature\n"
    )
    csv_path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))

    policy = ScoringPolicy(
        source_weights={"assay": 1.0, "literature": 0.5},
        min_support_threshold=0.5,
    )
    trace = csv_mod.evaluate_csv_observations(csv_path, policy=policy)

    assert trace.raw_support == pytest.approx(1.0)
    assert trace.raw_contradict == pytest.approx(0.0)
    assert trace.passed_threshold is True
    assert len(trace.contributions) == 2
    assert trace.contributions[0].observation_id == "obs-high"
    assert trace.contributions[0].signed_contribution == pytest.approx(1.0)
    assert trace.contributions[1].observation_id == "obs-zero"
    assert trace.contributions[1].signed_contribution == pytest.approx(0.0)


@pytest.mark.parametrize("bad_confidence", ["-0.01", "1.01", "nan", "not_a_number"])
def test_csv_row_rejects_out_of_range_or_non_finite_confidence(bad_confidence: str):
    csv_mod = _load_csv_module()
    row = {
        "observation_id": "obs-bad",
        "direction": "support",
        "confidence": bad_confidence,
        "source_type": "assay",
    }

    with pytest.raises(ValueError):
        csv_mod.csv_row_to_weighted_observation(row)


def test_csv_to_observations_with_bom_keeps_row_order(tmp_path: Path):
    csv_mod = _load_csv_module()
    csv_path = tmp_path / "panel_bom.csv"
    body = (
        "observation_id,entity,assay,log2FC,fdr,timepoint\n"
        "obs_first,VEGFA,rna_seq,2.2,0.001,12h\n"
        "obs_second,IL6,elisa,-2.1,0.002,12h\n"
    )
    csv_path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))

    rows = csv_mod.csv_to_observations(csv_path)
    assert len(rows) == 2
    assert rows[0]["observation_id"] == "obs_0001_VEGFA"
    assert rows[0]["entity"] == "VEGFA"
    assert rows[1]["observation_id"] == "obs_0002_IL6"
    assert rows[1]["entity"] == "IL6"


def test_infer_direction_exact_threshold_boundaries():
    from hypothesis_engine_v0_1.schemas import SignalDirection, ThresholdSpec

    csv_mod = _load_csv_module()
    th = ThresholdSpec()

    # Exact max_fdr boundary (fdr == th.max_fdr)
    at_max_fdr = csv_mod.infer_direction(th.min_abs_log2fc_up, th.max_fdr, th)
    assert at_max_fdr != SignalDirection.AMBIGUOUS

    # Exact unchanged upper/lower bounds (+max_abs_log2fc_unchanged and -max_abs_log2fc_unchanged)
    unchanged_pos = csv_mod.infer_direction(th.max_abs_log2fc_unchanged, 0.0, th)
    unchanged_neg = csv_mod.infer_direction(-th.max_abs_log2fc_unchanged, 0.0, th)
    assert unchanged_pos == SignalDirection.UNCHANGED
    assert unchanged_neg == SignalDirection.UNCHANGED

    # Exact up/down threshold bounds (+min_abs_log2fc_up and -min_abs_log2fc_up)
    up_exact = csv_mod.infer_direction(th.min_abs_log2fc_up, 0.0, th)
    down_exact = csv_mod.infer_direction(-th.min_abs_log2fc_up, 0.0, th)
    assert up_exact == SignalDirection.UP
    assert down_exact == SignalDirection.DOWN
