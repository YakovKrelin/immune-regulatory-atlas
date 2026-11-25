from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hypothesis_engine_v0_1 import cli as cli_module
from hypothesis_engine_v0_1.renderer import RenderReportResponse


def _sample_input_payload(query_id: str = "Q-CLI-TEST-001") -> dict:
    return {
        "query_id": query_id,
        "data_type": "purified_macrophage_culture",
        "cell_composition_changed": False,
        "timepoint_known": True,
        "target": {"entity": "IL6", "level": "mRNA"},
        "context": {
            "organism": "Homo sapiens",
            "cell_type": "Macrophage",
            "timepoint": "12h",
            "condition": {
                "oxygen": "Hypoxia",
                "oxygen_level": "1% O2",
                "duration": "12h",
            },
            "study_design": {"control": "Normoxia"},
        },
        "observations": [
            {
                "observation_id": "OBS-1",
                "entity": "VEGFA",
                "assay": "RNA-seq",
                "level": "mRNA",
                "direction": "up",
                "timepoint": "12h",
            },
            {
                "observation_id": "OBS-2",
                "entity": "SLC2A1",
                "assay": "RNA-seq",
                "level": "mRNA",
                "direction": "up",
                "timepoint": "12h",
            },
        ],
    }


def _utf8_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _write_utf8_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_bom_json(path: Path, payload: dict) -> Path:
    path.write_bytes(json.dumps(payload, indent=2).encode("utf-8-sig"))
    return path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_cli_reads_utf8_bom_and_prints_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_file = _write_bom_json(tmp_path / "bom_input.json", _sample_input_payload("Q-BOM-1"))

    exit_code = cli_module.main([str(input_file)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Q-BOM-1" in captured.out
    assert "HIF_PROGRAM" in captured.out
    assert captured.err == ""


def test_cli_json_flag_outputs_valid_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_file = _write_bom_json(tmp_path / "json_input.json", _sample_input_payload("Q-JSON-1"))

    exit_code = cli_module.main([str(input_file), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    parsed = json.loads(captured.out)
    assert parsed["engine_output"]["query_id"] == "Q-JSON-1"
    assert isinstance(parsed["markdown"], str)
    assert parsed["validation_passed"] is True
    assert parsed["violations"] == []
    assert len(parsed["input_hash"]) == 64
    assert len(parsed["output_hash"]) == 64


def test_cli_strict_and_no_strict_exit_codes_on_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_file = _write_bom_json(tmp_path / "violation_input.json", _sample_input_payload("Q-V-1"))

    def _fake_render(request, *, llm_client=None):
        return RenderReportResponse(
            markdown="IL6 is [removed_unsupported_causal_claim] HIF1A.",
            used_llm=False,
            validation_passed=False,
            violations=["Disallowed causal phrasing detected."],
            input_hash="a" * 64,
            output_hash="b" * 64,
            explanation_hash="c" * 64,
        )

    monkeypatch.setattr(cli_module, "render_hypothesis_report", _fake_render)

    strict_code = cli_module.main([str(input_file)])
    strict_captured = capsys.readouterr()
    assert strict_code == 1
    assert "Disallowed causal phrasing detected." in strict_captured.err

    non_strict_code = cli_module.main([str(input_file), "--no-strict"])
    non_strict_captured = capsys.readouterr()
    assert non_strict_code == 0
    assert "Disallowed causal phrasing detected." in non_strict_captured.err


def test_cli_missing_input_file_returns_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_file = tmp_path / "does_not_exist.json"

    exit_code = cli_module.main([str(missing_file)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "input file not found" in captured.err


def test_cli_subprocess_help_exits_0() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", "--help"],
        cwd=str(_repo_root()),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert proc.returncode == 0
    assert proc.stdout is not None
    assert "usage: hypothesis-engine-cli" in proc.stdout
    assert "--json" in proc.stdout
    assert "--max-bullets" in proc.stdout
    assert "--no-strict" in proc.stdout
    assert proc.stderr == ""


def test_cli_subprocess_markdown_and_json_output(tmp_path: Path) -> None:
    input_file = _write_utf8_json(tmp_path / "input.json", _sample_input_payload("Q-SUB-1"))

    md_proc = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", str(input_file)],
        cwd=str(_repo_root()),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert md_proc.returncode == 0
    assert md_proc.stdout is not None
    assert "Q-SUB-1" in md_proc.stdout
    assert "HIF_PROGRAM" in md_proc.stdout
    assert md_proc.stderr == ""

    json_proc = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", str(input_file), "--json", "--max-bullets", "2"],
        cwd=str(_repo_root()),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert json_proc.returncode == 0
    assert json_proc.stdout is not None
    parsed = json.loads(json_proc.stdout)
    assert parsed["engine_output"]["query_id"] == "Q-SUB-1"
    assert parsed["validation_passed"] is True
    assert parsed["violations"] == []


def test_cli_subprocess_missing_file_exits_2(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    proc = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", str(missing_path)],
        cwd=str(_repo_root()),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert proc.stderr is not None
    assert "input file not found" in proc.stderr


def test_cli_subprocess_invalid_schema_exits_nonzero(tmp_path: Path) -> None:
    invalid_file = tmp_path / "invalid_schema.json"
    invalid_payload = {
        "query_id": "Q-INVALID-SCHEMA-1",
        "data_type": "purified_macrophage_culture",
        "cell_composition_changed": "not_a_boolean",
    }
    _write_utf8_json(invalid_file, invalid_payload)

    proc = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", str(invalid_file)],
        cwd=str(_repo_root()),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert proc.returncode != 0
    assert proc.stdout == ""
    assert proc.stderr is not None
    assert "ValidationError" in proc.stderr or "validation error" in proc.stderr.lower()


def test_cli_subprocess_malformed_json_exits_nonzero(tmp_path: Path) -> None:
    malformed_file = tmp_path / "malformed.json"
    malformed_file.write_text('{"query_id": "Q-BAD-SYNTAX-1", "observations": [}', encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", str(malformed_file)],
        cwd=str(_repo_root()),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert proc.returncode != 0
    assert proc.stdout == ""
    assert proc.stderr is not None
    assert "jsondecodeerror" in proc.stderr.lower() or "expecting" in proc.stderr.lower()


def test_cli_subprocess_invalid_args_exits_2(tmp_path: Path) -> None:
    input_file = _write_utf8_json(tmp_path / "input.json", _sample_input_payload("Q-ARGS-1"))

    bad_flag = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", str(input_file), "--nonexistent-flag"],
        cwd=str(_repo_root()),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert bad_flag.returncode == 2
    assert bad_flag.stdout == ""
    assert bad_flag.stderr is not None
    assert "unrecognized arguments" in bad_flag.stderr.lower()

    bad_int = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", str(input_file), "--max-bullets", "abc"],
        cwd=str(_repo_root()),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert bad_int.returncode == 2
    assert bad_int.stdout == ""
    assert bad_int.stderr is not None
    assert "invalid int value" in bad_int.stderr.lower()


def test_cli_subprocess_no_strict_exits_0_on_violations(tmp_path: Path) -> None:
    input_file = _write_utf8_json(tmp_path / "input.json", _sample_input_payload("Q-NO-STRICT-1"))

    wrapper_script = tmp_path / "run_cli_with_violation.py"
    wrapper_script.write_text(
        """
from __future__ import annotations

import sys

from hypothesis_engine_v0_1 import cli as cli_module
from hypothesis_engine_v0_1.renderer import RenderReportResponse


def _fake_render(request, *, llm_client=None):
    return RenderReportResponse(
        markdown="Hypothesis Engine Summary: Q-NO-STRICT-1 [removed_unsupported_causal_claim]",
        used_llm=False,
        validation_passed=False,
        violations=["Disallowed causal phrasing detected."],
        input_hash="a" * 64,
        output_hash="b" * 64,
        explanation_hash="c" * 64,
    )


cli_module.render_hypothesis_report = _fake_render
raise SystemExit(cli_module.main(sys.argv[1:]))
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(wrapper_script), str(input_file), "--no-strict"],
        cwd=str(_repo_root()),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert proc.returncode == 0
    assert proc.stdout is not None
    assert "Q-NO-STRICT-1" in proc.stdout
    assert proc.stderr is not None
    assert "Disallowed causal phrasing detected." in proc.stderr

def test_schemas_resolve_direction_and_panel_validators() -> None:
    from pydantic import ValidationError
    from hypothesis_engine_v0_1 import schemas

    assert schemas.Observation(
        observation_id="O1", entity="IL6", assay="rna_seq", direction=None, unit="log2FC", value=None
    ).resolve_direction() == schemas.SignalDirection.NOT_MEASURED

    assert schemas.Observation(
        observation_id="O2", entity="IL6", assay="rna_seq", direction=None, unit="log2FC", value=2.5, fdr=0.5
    ).resolve_direction() == schemas.SignalDirection.AMBIGUOUS

    thresh = schemas.ThresholdSpec(min_abs_log2fc_up=2.0, max_abs_log2fc_unchanged=0.5, max_fdr=0.1)
    assert schemas.Observation(
        observation_id="O3", entity="IL6", assay="rna_seq", direction=None, unit="log2FC", value=2.5, fdr=0.01, threshold=thresh
    ).resolve_direction() == schemas.SignalDirection.UP
    assert schemas.Observation(
        observation_id="O4", entity="IL6", assay="rna_seq", direction=None, unit="log2FC", value=-2.5, fdr=0.01, threshold=thresh
    ).resolve_direction() == schemas.SignalDirection.DOWN
    assert schemas.Observation(
        observation_id="O5", entity="IL6", assay="rna_seq", direction=None, unit="log2FC", value=0.2, fdr=0.01, threshold=thresh
    ).resolve_direction() == schemas.SignalDirection.UNCHANGED
    assert schemas.Observation(
        observation_id="O6", entity="IL6", assay="rna_seq", direction=None, unit="log2FC", value=1.2, fdr=0.01, threshold=thresh
    ).resolve_direction() == schemas.SignalDirection.AMBIGUOUS
    assert schemas.Observation(
        observation_id="O7", entity="IL6", assay="rna_seq", direction=None, unit="TPM", value=12.5
    ).resolve_direction() == schemas.SignalDirection.NOT_MEASURED

    base_inp = schemas.HypothesisEngineInput(
        query_id="Q-SCHEMA",
        target=schemas.TargetSpec(entity="IL6", level=schemas.MeasurementLevel.MRNA),
        context=schemas.BiologicalContext(
            organism="Homo sapiens",
            cell_type="macrophage",
            study_design=schemas.StudyDesignContext(control="ctrl"),
        ),
    )

    empty_panel = schemas.ProgramPanel(program_id="EMPTY_P", core_markers=[], supporting_markers=[])
    with pytest.raises((ValueError, ValidationError), match="at least one marker"):
        schemas.HypothesisEngineInput.model_validate(
            {
                **base_inp.model_dump(mode="python"),
                "hif_panel": empty_panel.model_dump(mode="python"),
            }
        )

    overlap_panel = schemas.ProgramPanel(
        program_id="ORTH_P",
        core_markers=list(base_inp.hif_panel.core_markers[:1]),
        supporting_markers=[],
    )
    with pytest.raises((ValueError, ValidationError), match="overlaps with H1/H2 markers"):
        schemas.HypothesisEngineInput.model_validate(
            {
                **base_inp.model_dump(mode="python"),
                "orthogonal_state_panel": overlap_panel.model_dump(mode="python"),
            }
        )


def test_cli_module_dunder_main_execution() -> None:
    env = _utf8_env()
    cov_cfg = _repo_root() / "pyproject.toml"
    if cov_cfg.exists():
        env["COVERAGE_PROCESS_START"] = str(cov_cfg)
    proc = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", "--help"],
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0