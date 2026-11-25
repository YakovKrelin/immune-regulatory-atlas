from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _sample_input_payload(query_id: str = "Q-SUB-001") -> dict:
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


def test_cli_subprocess_markdown_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    input_file = _write_utf8_json(tmp_path / "input.json", _sample_input_payload("Q-SUB-MD-1"))

    proc = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", str(input_file)],
        cwd=str(repo_root),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert proc.returncode == 0
    assert proc.stdout is not None
    assert "Q-SUB-MD-1" in proc.stdout
    assert "HIF_PROGRAM" in proc.stdout
    assert proc.stderr == ""


def test_cli_subprocess_json_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    input_file = _write_utf8_json(tmp_path / "input.json", _sample_input_payload("Q-SUB-JSON-1"))

    proc = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", str(input_file), "--json"],
        cwd=str(repo_root),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert proc.returncode == 0
    assert proc.stdout is not None
    parsed = json.loads(proc.stdout)
    assert parsed["engine_output"]["query_id"] == "Q-SUB-JSON-1"
    assert parsed["validation_passed"] is True
    assert parsed["violations"] == []
    assert len(parsed["input_hash"]) == 64
    assert len(parsed["output_hash"]) == 64


def test_cli_subprocess_missing_file_exits_2(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    missing_path = tmp_path / "missing.json"

    proc = subprocess.run(
        [sys.executable, "-m", "hypothesis_engine_v0_1.cli", str(missing_path)],
        cwd=str(repo_root),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert proc.returncode == 2
    assert proc.stderr is not None
    assert "input file not found" in proc.stderr