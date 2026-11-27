from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest


def _load_csv_module():
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "csv_to_observations.py"
    assert module_path.exists(), f"Missing {module_path}"
    spec = importlib.util.spec_from_file_location("csv_to_observations", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, text: str = "", encoding: str = "utf-8", raw_bytes: bytes | None = None):
    if raw_bytes is not None:
        path.write_bytes(raw_bytes)
    else:
        path.write_text(text, encoding=encoding)


@pytest.mark.parametrize(
    "loader_name",
    ["load_weighted_observations_from_csv", "csv_to_observations"],
)
def test_rejects_empty_file(tmp_path: Path, loader_name: str):
    mod = _load_csv_module()
    fn = getattr(mod, loader_name)
    csv_file = tmp_path / "empty.csv"
    _write_csv(csv_file, "")

    with pytest.raises(ValueError):
        fn(csv_file)


def test_header_only_weighted_file_returns_empty_list(tmp_path: Path):
    mod = _load_csv_module()
    csv_file = tmp_path / "header_only.csv"
    _write_csv(csv_file, "observation_id,direction,confidence,source_type\n")

    assert mod.load_weighted_observations_from_csv(csv_file) == []


@pytest.mark.parametrize(
    "bad_header,row",
    [
        ("direction,confidence,source_type", "support,0.8,transcriptomics"),
        ("observation_id,confidence,source_type", "obs-1,0.8,transcriptomics"),
        ("observation_id,direction,source_type", "obs-1,support,transcriptomics"),
        ("observation_id,direction,confidence", "obs-1,support,0.8"),
    ],
)
def test_rejects_missing_required_columns_weighted(tmp_path: Path, bad_header: str, row: str):
    mod = _load_csv_module()
    csv_file = tmp_path / "missing_col.csv"
    _write_csv(csv_file, f"{bad_header}\n{row}\n")

    with pytest.raises(ValueError):
        mod.load_weighted_observations_from_csv(csv_file)


@pytest.mark.parametrize(
    "bad_row",
    [
        "obs-1,support,not_a_number,transcriptomics",
        "obs-1,support,NaN,transcriptomics",
        "obs-1,support,inf,transcriptomics",
        "obs-1,invalid_direction,0.7,transcriptomics",
        "obs-1,support,1.5,transcriptomics",
        "obs-1,support,-0.2,transcriptomics",
        ",support,0.7,transcriptomics",
        "obs-2,support,,transcriptomics",
        "obs-3,support,0.7,",
    ],
)
def test_rejects_invalid_types_bounds_and_blank_fields_weighted(tmp_path: Path, bad_row: str):
    mod = _load_csv_module()
    csv_file = tmp_path / "bad_row.csv"
    _write_csv(
        csv_file,
        f"observation_id,direction,confidence,source_type\n{bad_row}\n",
    )

    with pytest.raises((ValueError, TypeError)):
        mod.load_weighted_observations_from_csv(csv_file)


def test_rejects_duplicate_observation_id(tmp_path: Path):
    mod = _load_csv_module()
    csv_file = tmp_path / "dup_id.csv"
    _write_csv(
        csv_file,
        "observation_id,direction,confidence,source_type\n"
        "obs-dup,support,0.75,transcriptomics\n"
        "obs-dup,support,0.65,transcriptomics\n",
    )

    with pytest.raises(ValueError, match="Duplicate observation_id"):
        mod.load_weighted_observations_from_csv(csv_file)


def test_rejects_conflicting_duplicate_observation_id(tmp_path: Path):
    mod = _load_csv_module()
    csv_file = tmp_path / "conflict_dup.csv"
    _write_csv(
        csv_file,
        "observation_id,direction,confidence,source_type\n"
        "obs-1,support,0.80,transcriptomics\n"
        "obs-1,contradict,0.80,transcriptomics\n",
    )

    with pytest.raises(ValueError, match="Duplicate observation_id"):
        mod.load_weighted_observations_from_csv(csv_file)


def test_missing_entity_column_returns_empty_and_blank_entity_raises(tmp_path: Path):
    mod = _load_csv_module()
    bad_col_file = tmp_path / "no_entity_col.csv"
    _write_csv(bad_col_file, "log2FC,fdr\n1.5,0.01\n")
    assert mod.csv_to_observations(bad_col_file) == []

    blank_entity_file = tmp_path / "blank_entity.csv"
    _write_csv(blank_entity_file, "entity,log2FC,fdr\n,1.5,0.01\n")
    with pytest.raises(ValueError, match="entity"):
        mod.csv_to_observations(blank_entity_file)


def test_rejects_invalid_encoding_bytes(tmp_path: Path):
    mod = _load_csv_module()
    csv_file = tmp_path / "invalid_encoding.csv"
    _write_csv(csv_file, raw_bytes=b"observation_id,direction,confidence,source_type\nobs-1,\xff\xfe\x00\x00,0.5,assay\n")

    with pytest.raises(UnicodeDecodeError):
        mod.load_weighted_observations_from_csv(csv_file)
