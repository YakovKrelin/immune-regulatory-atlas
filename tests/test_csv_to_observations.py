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


def _enum_val(val):
    return getattr(val, "value", val)


def _build_nested_model(model_cls, values: dict):
    if model_cls is None:
        return {k: v for k, v in values.items() if v is not None}
    fields = getattr(model_cls, "model_fields", {})
    kwargs = {}
    for k, v in values.items():
        if k in fields and v is not None:
            kwargs[k] = v
    return model_cls(**kwargs)


def _make_context(
    schemas,
    *,
    organism="Homo sapiens",
    oxygen="Hypoxia",
    oxygen_level="1% O2",
    cell_type="macrophage",
    tissue="myeloid_culture",
    timepoint="12h",
    control="normoxia_21pct_o2",
):
    ctx_cls = getattr(schemas, "BiologicalContext", None) or getattr(schemas, "ContextSpec", None)
    cond_cls = getattr(schemas, "ConditionContext", None)
    study_cls = getattr(schemas, "StudyDesignContext", None)

    cond_obj = _build_nested_model(
        cond_cls,
        {
            "oxygen": oxygen,
            "oxygen_level": oxygen_level,
        },
    )
    study_obj = _build_nested_model(
        study_cls,
        {
            "control": control,
        },
    )

    top_values = {
        "organism": organism,
        "cell_type": cell_type,
        "tissue": tissue,
        "timepoint": timepoint,
        "oxygen": oxygen,
        "oxygen_level": oxygen_level,
        "condition": cond_obj,
        "study_design": study_obj,
    }
    return _build_nested_model(ctx_cls, top_values)


def _build_orthogonal_panel(schemas, entities=("MRC1", "CD163", "MERTK")):
    panel_cls = (
        getattr(schemas, "OrthogonalStatePanel", None)
        or getattr(schemas, "ProgramPanel", None)
        or getattr(schemas, "MarkerPanel", None)
        or getattr(schemas, "GenePanel", None)
    )
    if panel_cls is None:
        return None
    fields = getattr(panel_cls, "model_fields", {})
    kwargs = {}
    for name, field_info in fields.items():
        lname = name.lower()
        if "id" in lname and "program" in lname:
            kwargs[name] = "ORTHOGONAL_MACROPHAGE_STATE_PROGRAM"
        elif name == "program_id":
            kwargs[name] = "ORTHOGONAL_MACROPHAGE_STATE_PROGRAM"
        elif "name" in lname:
            kwargs[name] = "Orthogonal macrophage functional-state panel"
        elif any(tok in lname for tok in ("entities", "markers", "genes", "items")):
            ann = getattr(field_info, "annotation", None)
            if ann is not None and "dict" in str(ann).lower():
                kwargs[name] = {ent: "increase" for ent in entities}
            else:
                kwargs[name] = list(entities)
    try:
        return panel_cls(**kwargs)
    except Exception:
        return None


def _build_payload(
    schemas,
    *,
    query_id,
    scenario,
    target_entity="IL6",
    level=None,
    context=None,
    observations=None,
    orthogonal_state_panel=None,
):
    target_cls = getattr(schemas, "TargetSpec", None)
    level_val = level if level is not None else getattr(getattr(schemas, "MeasurementLevel", None), "MRNA", "mRNA")
    target = target_cls(entity=target_entity, level=level_val) if target_cls else {"entity": target_entity, "level": _enum_val(level_val)}

    input_cls = getattr(schemas, "HypothesisEngineInput", None)
    kwargs = {
        "query_id": query_id,
        "scenario": scenario,
        "target": target,
        "context": context,
        "observations": observations or [],
    }
    if input_cls is not None:
        in_fields = getattr(input_cls, "model_fields", {})
        if "cell_composition_changed" in in_fields:
            kwargs["cell_composition_changed"] = False
        if "timepoint_known" in in_fields:
            kwargs["timepoint_known"] = True
        if orthogonal_state_panel is not None and "orthogonal_state_panel" in in_fields:
            kwargs["orthogonal_state_panel"] = orthogonal_state_panel
        return input_cls(**kwargs)

    if orthogonal_state_panel is not None:
        kwargs["orthogonal_state_panel"] = orthogonal_state_panel
    return kwargs


def test_csv_e2e_evaluates_h1_to_h4_with_valid_context(tmp_path: Path):
    from hypothesis_engine_v0_1 import engine as eng, schemas

    csv_mod = _load_csv_module()
    csv_file = tmp_path / "h1_h4_panel.csv"
    csv_file.write_text(
        "observation_id,entity,assay,log2FC,fdr,timepoint\n"
        "obs_1,VEGFA,rna_seq,2.5,0.001,12h\n"
        "obs_2,SLC2A1,rna_seq,2.0,0.002,12h\n"
        "obs_3,P65_PHOSPHO,immunoblot,2.2,0.004,12h\n"
        "obs_4,IKBA_DEGRADATION,immunoblot,1.9,0.005,12h\n"
        "obs_5,NFKBIA,rna_seq,2.1,0.001,12h\n"
        "obs_6,TNFAIP3,rna_seq,2.3,0.001,12h\n"
        "obs_7,HIF1A,rna_seq,1.8,0.003,12h\n"
        "obs_8,IL6,elisa,2.6,0.001,12h\n"
        "obs_9,MRC1,rna_seq,2.1,0.002,12h\n"
        "obs_10,CD163,rna_seq,1.9,0.003,12h\n"
        "obs_11,MERTK,rna_seq,2.0,0.002,12h\n",
        encoding="utf-8",
    )

    observations = csv_mod.csv_to_observations(csv_file)
    assert len(observations) == 11

    ctx = _make_context(
        schemas,
        organism="Homo sapiens",
        oxygen="Hypoxia",
        oxygen_level="1% O2",
        cell_type="macrophage",
        control="normoxia_21pct_o2",
    )
    orth_panel = _build_orthogonal_panel(schemas, entities=("MRC1", "CD163", "MERTK"))
    bulk_dt = getattr(getattr(schemas, "DataType", None), "BULK_RNA_SEQ", "bulk_rna_seq")

    def _run_with_comp_flag(comp_changed: bool):
        pl = _build_payload(
            schemas,
            query_id=f"e2e_h1_h4_{comp_changed}",
            scenario="IL6_hypoxia_macrophage_panel",
            target_entity="IL6",
            context=ctx,
            observations=observations,
            orthogonal_state_panel=orth_panel,
        )
        if hasattr(pl, "data_type"):
            pl.data_type = bulk_dt
        elif isinstance(pl, dict):
            pl["data_type"] = bulk_dt
        if hasattr(pl, "cell_composition_changed"):
            pl.cell_composition_changed = comp_changed
        elif isinstance(pl, dict):
            pl["cell_composition_changed"] = comp_changed
        return eng.run_hypothesis_engine(pl)

    def _is_eval(h_eval) -> bool:
        ev_status = str(_enum_val(h_eval.evidence_status)).lower()
        dec_status = str(_enum_val(h_eval.decision_status)).lower()
        data_status = str(_enum_val(getattr(h_eval, "data_status", ""))).lower()
        applied_rules = getattr(h_eval, "applied_rules", None) or []
        return (
            ev_status != "not_assessed"
            or dec_status != "not_assessed"
            or data_status not in {"", "not_assessed"}
            or len(applied_rules) > 0
        )

    out_no_shift = _run_with_comp_flag(False)
    assert str(_enum_val(out_no_shift.mode)).lower() != "hard_abstention"
    for h_key in ("H1", "H2", "H3"):
        assert h_key in out_no_shift.hypotheses
        assert _is_eval(out_no_shift.hypotheses[h_key]), f"{h_key} unassessed: {out_no_shift.hypotheses[h_key]}"

    out_with_shift = _run_with_comp_flag(True)
    assert str(_enum_val(out_with_shift.mode)).lower() != "hard_abstention"
    assert "H4" in out_with_shift.hypotheses
    assert _is_eval(out_with_shift.hypotheses["H4"]), f"H4 unassessed: {out_with_shift.hypotheses['H4']}"


def test_csv_e2e_detects_context_conflict_and_counter_sign_signals(tmp_path: Path):
    from hypothesis_engine_v0_1 import engine as eng, schemas

    csv_mod = _load_csv_module()
    csv_file = tmp_path / "conflict_obs.csv"
    csv_file.write_text(
        "observation_id,entity,assay,log2FC,fdr,timepoint\n"
        "obs_1,VEGFA,rna_seq,-2.3,0.001,12h\n"
        "obs_2,SLC2A1,rna_seq,-1.9,0.002,12h\n"
        "obs_3,P65_PHOSPHO,immunoblot,2.1,0.003,12h\n"
        "obs_4,NFKBIA,rna_seq,-1.8,0.002,12h\n"
        "obs_5,IL6,elisa,2.4,0.001,12h\n",
        encoding="utf-8",
    )

    observations = csv_mod.csv_to_observations(csv_file)
    assert len(observations) == 5

    ctx = _make_context(
        schemas,
        organism="Homo sapiens",
        oxygen="Normoxia",
        oxygen_level="21% O2",
        cell_type="macrophage",
        control="untreated_control",
    )
    payload = _build_payload(
        schemas,
        query_id="e2e_conflict_002",
        scenario="normoxia_21pct_conflict",
        target_entity="IL6",
        context=ctx,
        observations=observations,
    )
    output = eng.run_hypothesis_engine(payload)

    mode = str(_enum_val(output.mode)).lower()
    assert mode != "hard_abstention", f"Unexpected abstention on conflict case: {output}"

    conflicts = getattr(output, "context_conflicts", None) or []
    rule_conflicts = getattr(output, "rule_conflicts", None) or []
    has_evidence_conflict = any(
        bool(getattr(h, "evidence_conflict", False))
        or bool(getattr(h, "rule_conflict", False))
        or (len(getattr(h, "supporting_evidence", []) or []) > 0 and len(getattr(h, "contradicting_evidence", []) or []) > 0)
        for h in output.hypotheses.values()
    )

    assert mode == "weak_inference" or len(conflicts) > 0 or len(rule_conflicts) > 0 or has_evidence_conflict, (
        f"Expected conflict or weak_inference signal, got mode={output.mode}, "
        f"context_conflicts={conflicts}, rule_conflicts={rule_conflicts}"
    )


def test_csv_e2e_triggers_hard_abstention_when_study_design_control_missing(tmp_path: Path):
    from hypothesis_engine_v0_1 import engine as eng, schemas

    csv_mod = _load_csv_module()
    csv_file = tmp_path / "missing_control_obs.csv"
    csv_file.write_text(
        "observation_id,entity,assay,log2FC,fdr,timepoint\n"
        "obs_1,VEGFA,rna_seq,2.2,0.001,12h\n"
        "obs_2,IL6,elisa,2.0,0.002,12h\n",
        encoding="utf-8",
    )

    observations = csv_mod.csv_to_observations(csv_file)
    assert len(observations) == 2

    ctx_no_control = _make_context(
        schemas,
        organism="Homo sapiens",
        oxygen="Hypoxia",
        oxygen_level="1% O2",
        cell_type="macrophage",
        control=None,
    )
    payload = _build_payload(
        schemas,
        query_id="e2e_abstain_003",
        scenario="missing_control_abstention",
        target_entity="IL6",
        context=ctx_no_control,
        observations=observations,
    )
    output = eng.run_hypothesis_engine(payload)

    mode = str(_enum_val(output.mode)).lower()
    missing_mandatory = getattr(output, "missing_mandatory_fields", None) or []
    missing_context = getattr(output, "missing_context_fields", None) or []
    control_flagged = any("control" in str(field).lower() for field in (*missing_mandatory, *missing_context))

    assert control_flagged, (
        f"Expected control field to be flagged when study_design.control is None: "
        f"mode={output.mode}, missing_mandatory={missing_mandatory}, missing_context={missing_context}"
    )

    if any("control" in str(field).lower() for field in missing_mandatory):
        assert mode == "hard_abstention", f"Expected hard_abstention for mandatory control, got mode={output.mode}"
    else:
        assert mode in {"weak_inference", "hard_abstention"}, f"Unexpected mode={output.mode}"


def test_csv_missing_required_columns_fails_or_skips_rows(tmp_path: Path):
    csv_mod = _load_csv_module()
    csv_file = tmp_path / "missing_cols.csv"
    csv_file.write_text(
        "observation_id,assay,timepoint\n"
        "obs_1,rna_seq,12h\n",
        encoding="utf-8",
    )

    try:
        rows = csv_mod.csv_to_observations(csv_file)
    except (KeyError, ValueError):
        return

    assert isinstance(rows, list)
    assert len(rows) == 0 or all(
        not getattr(r, "entity", None) and getattr(r, "value", None) is None
        for r in rows
    )


def test_csv_empty_and_nan_log2fc_and_fdr_handled_safely(tmp_path: Path):
    import math
    csv_mod = _load_csv_module()
    csv_file = tmp_path / "nan_and_empty_nums.csv"
    csv_file.write_text(
        "observation_id,entity,assay,log2FC,fdr,timepoint\n"
        "obs_1,VEGFA,rna_seq,,0.001,12h\n"
        "obs_2,SLC2A1,rna_seq,NaN,NaN,12h\n"
        "obs_3,IL6,elisa,2.4,,12h\n",
        encoding="utf-8",
    )

    def _get_field(row, *names):
        if isinstance(row, dict):
            for n in names:
                if n in row:
                    return row[n]
            return None
        for n in names:
            if hasattr(row, n):
                return getattr(row, n)
        return None

    rows = csv_mod.csv_to_observations(csv_file)
    assert len(rows) == 3

    for r in rows:
        val = _get_field(r, "value", "log2fc", "log2FC")
        fdr_val = _get_field(r, "fdr")
        if isinstance(val, float) and not math.isnan(val):
            assert math.isfinite(val)
        if isinstance(fdr_val, float) and not math.isnan(fdr_val):
            assert math.isfinite(fdr_val)

    obs_1, obs_2, obs_3 = rows
    assert _get_field(obs_1, "value", "log2fc", "log2FC") is None
    assert _get_field(obs_2, "value", "log2fc", "log2FC") is None
    assert _get_field(obs_2, "fdr") is None
    assert _get_field(obs_3, "value", "log2fc", "log2FC") == 2.4
    assert _get_field(obs_3, "fdr") is None


def test_csv_unknown_measurement_level_falls_back_or_raises(tmp_path: Path):
    from hypothesis_engine_v0_1 import schemas

    csv_mod = _load_csv_module()
    csv_file = tmp_path / "bad_level.csv"
    csv_file.write_text(
        "observation_id,entity,assay,level,log2FC,fdr,timepoint\n"
        "obs_1,IL6,elisa,unknown_alien_level,2.2,0.001,12h\n",
        encoding="utf-8",
    )

    default_level = getattr(getattr(schemas, "MeasurementLevel", None), "MRNA", "mRNA")
    try:
        rows = csv_mod.csv_to_observations(csv_file, default_level=default_level)
    except (ValueError, KeyError):
        return

    assert len(rows) == 1
    obs_level = getattr(rows[0], "level", None)
    assert obs_level is not None
    assert _enum_val(obs_level) == _enum_val(default_level) or str(_enum_val(obs_level)).lower() in {
        "unknown_alien_level",
        "not_measured",
        "other",
    }
