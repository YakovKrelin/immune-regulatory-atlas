import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_SCRIPT_PATH = REPO_ROOT / "scripts" / "csv_to_observations.py"


def _load_csv_module():
    spec = importlib.util.spec_from_file_location("csv_to_observations_script", CSV_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_float_handles_valid_and_invalid_values():
    csv_mod = _load_csv_module()
    assert csv_mod._parse_float("1.75") == pytest.approx(1.75)
    assert csv_mod._parse_float("-0.42") == pytest.approx(-0.42)
    assert csv_mod._parse_float("") is None
    assert csv_mod._parse_float("not_a_number") is None
    assert csv_mod._parse_float(None) is None


def test_infer_direction_respects_threshold_and_fdr():
    csv_mod = _load_csv_module()
    threshold = csv_mod.ThresholdSpec(min_abs_log2fc=1.0, max_fdr=0.05)

    assert csv_mod.infer_direction(2.0, 0.01, threshold).value.lower() == "up"
    assert csv_mod.infer_direction(-2.0, 0.01, threshold).value.lower() == "down"
    assert csv_mod.infer_direction(0.1, 0.01, threshold).value.lower() == "unchanged"


def test_csv_to_observations_and_build_engine_input(tmp_path: Path):
    csv_mod = _load_csv_module()
    sample_csv = tmp_path / "hypoxia_obs.csv"
    sample_csv.write_text(
        "entity,log2fc,fdr,timepoint,measurement_level\n"
        "IL6,2.4,0.005,12h,mRNA\n"
        "VEGFA,1.9,0.010,12h,mRNA\n"
        "LDHA,-1.8,0.020,12h,mRNA\n",
        encoding="utf-8",
    )

    observations = csv_mod.csv_to_observations(sample_csv)
    assert len(observations) == 3

    engine_input = csv_mod.build_engine_input(sample_csv, query_id="CSV-REGR-001")
    assert engine_input is not None


def test_csv_observations_end_to_end_run_hypothesis_engine(tmp_path: Path):
    from hypothesis_engine_v0_1 import engine as eng, schemas

    csv_mod = _load_csv_module()
    csv_file = tmp_path / "hypoxia_nfkb_obs.csv"
    csv_file.write_text(
        "observation_id,entity,assay,log2FC,fdr,timepoint\n"
        "obs_1,VEGFA,rna_seq,2.4,0.001,12h\n"
        "obs_2,SLC2A1,rna_seq,1.9,0.002,12h\n"
        "obs_3,P65_PHOSPHO,immunoblot,2.2,0.005,12h\n"
        "obs_4,IKBA_DEGRADATION,immunoblot,1.8,0.005,12h\n"
        "obs_5,NFKBIA,rna_seq,2.0,0.001,12h\n"
        "obs_6,TNFAIP3,rna_seq,2.1,0.001,12h\n",
        encoding="utf-8",
    )

    converter = getattr(csv_mod, "csv_to_observations", None) or getattr(csv_mod, "convert_csv_to_observations")
    observations = converter(csv_file)
    assert len(observations) == 6

    def _dir_val(item):
        raw = item["direction"] if isinstance(item, dict) else item.direction
        return getattr(raw, "value", raw)

    assert all(_dir_val(obs) == "up" for obs in observations)

    payload = schemas.HypothesisEngineInput(
        query_id="e2e_csv_001",
        scenario="IL6_hypoxia_macrophage",
        target=schemas.TargetSpec(entity="IL6", level=schemas.MeasurementLevel.MRNA),
        context=schemas.BiologicalContext(
            organism="Mus musculus",
            cell_type="macrophage",
            condition=schemas.ConditionContext(oxygen="Hypoxia", oxygen_level="1% O2"),
            study_design=schemas.StudyDesignContext(control="normoxia_control"),
        ),
        observations=observations,
    )
    output = eng.run_hypothesis_engine(payload)
    assert getattr(output.mode, "value", output.mode) == "weak_inference", (
        f"mode={output.mode}, missing_mandatory={getattr(output, 'missing_mandatory_fields', None)}, "
        f"missing_context={getattr(output, 'missing_context_fields', None)}, "
        f"context_conflicts={getattr(output, 'context_conflicts', None)}"
    )
    assert getattr(output.hypotheses["H1"].data_status, "value", output.hypotheses["H1"].data_status) == "compatible"
