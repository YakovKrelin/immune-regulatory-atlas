import json
import runpy
from pathlib import Path

import pytest

from scripts.score_observations import main, run_cli


def test_run_cli_success_with_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_file = tmp_path / "payload.json"
    input_file.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "observation_id": "obs-1",
                        "direction": "support",
                        "confidence": 0.9,
                        "source_type": "assay",
                    }
                ],
                "policy": {
                    "source_weights": {"assay": 1.5},
                    "min_support_threshold": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = run_cli(["--input", str(input_file)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    output = json.loads(captured.out)
    assert output["passed_threshold"] is True
    assert output["raw_support"] == pytest.approx(1.35)


def test_run_cli_returns_error_on_missing_or_invalid_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_file = tmp_path / "does_not_exist.json"
    code_missing = run_cli(["--input", str(missing_file)])
    err_missing = capsys.readouterr().err
    assert code_missing == 1
    assert "Error loading scoring input:" in err_missing

    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text('{"observations": [{"confidence": 5.0}]}', encoding="utf-8")
    code_invalid = run_cli(["--input", str(invalid_file)])
    err_invalid = capsys.readouterr().err
    assert code_invalid == 1
    assert "Error loading scoring input:" in err_invalid


def test_main_exits_with_run_cli_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "empty.json"
    input_file.write_text('{"observations": []}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["score_observations.py", "--input", str(input_file)])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0


def test_script_dunder_main_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "dunder_main.json"
    input_file.write_text('{"observations": []}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["score_observations.py", "--input", str(input_file)])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path("scripts/score_observations.py", run_name="__main__")

    assert exc_info.value.code == 0
