from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from hypothesis_engine_v0_1.schemas import MeasurementLevel, Observation, SignalDirection, ThresholdSpec
from hypothesis_engine_v0_1.scoring_rules import (
    ScoringPolicy,
    ScoringTrace,
    WeightedObservation,
    evaluate_weighted_hypothesis,
)

REQUIRED_WEIGHTED_COLUMNS = ("observation_id", "direction", "confidence", "source_type")
ALLOWED_WEIGHTED_DIRECTIONS = {"support", "contradict"}


def infer_direction(
    log2fc: float | None,
    fdr: float | None,
    threshold: ThresholdSpec,
) -> SignalDirection:
    if log2fc is None:
        return SignalDirection.NOT_MEASURED
    if fdr is not None and fdr > threshold.max_fdr:
        return SignalDirection.AMBIGUOUS
    if log2fc >= threshold.min_abs_log2fc_up:
        return SignalDirection.UP
    if log2fc <= -threshold.min_abs_log2fc_up:
        return SignalDirection.DOWN
    if abs(log2fc) <= threshold.max_abs_log2fc_unchanged:
        return SignalDirection.UNCHANGED
    return SignalDirection.AMBIGUOUS


def _parse_float(raw: str | None, *, field_name: str = "value", strict: bool = False) -> float | None:
    if raw is None:
        if strict:
            raise ValueError(f"Invalid {field_name} value: {raw!r}")
        return None
    val = raw.strip()
    if not val or val.lower() in {"na", "nan", "none", "null"}:
        if strict:
            raise ValueError(f"Invalid {field_name} value: {raw!r}")
        return None
    try:
        parsed = float(val)
    except (ValueError, TypeError) as exc:
        if strict:
            raise ValueError(f"Invalid {field_name} value: {raw!r}") from exc
        return None
    if not math.isfinite(parsed):
        if strict:
            raise ValueError(f"Invalid {field_name} value: {raw!r}")
        return None
    return parsed


def _parse_level(raw: str | None, fallback: MeasurementLevel) -> MeasurementLevel:
    if raw is None or not raw.strip():
        return fallback
    val = raw.strip()
    try:
        return MeasurementLevel(val)
    except ValueError:
        upper = val.upper()
        if upper in MeasurementLevel.__members__:
            return MeasurementLevel[upper]
        raise ValueError(f"Invalid measurement level: {raw!r}")


def csv_to_observations(
    csv_path: str | Path,
    *,
    default_assay: str = "RNA-seq",
    default_level: MeasurementLevel = MeasurementLevel.MRNA,
    threshold: ThresholdSpec | None = None,
) -> list[dict[str, Any]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    th = threshold or ThresholdSpec()
    observations: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or missing header row")
        fieldnames = {name.strip() for name in reader.fieldnames if name is not None}
        if "entity" not in fieldnames:
            return []

        for idx, row in enumerate(reader, start=1):
            entity = (row.get("entity") or "").strip()
            if not entity:
                raise ValueError(f"Row {idx} has empty required field 'entity'")

            log2fc = _parse_float(row.get("log2FC"), field_name="log2FC", strict=False)
            fdr = _parse_float(row.get("fdr"), field_name="fdr", strict=False)
            direction = infer_direction(log2fc, fdr, th)
            level = _parse_level(row.get("level"), default_level)

            obs = Observation(
                observation_id=f"obs_{idx:04d}_{entity}",
                entity=entity,
                assay=(row.get("assay") or default_assay).strip(),
                level=level,
                direction=direction,
                value=log2fc,
                unit="log2FC" if log2fc is not None else None,
                fdr=fdr,
                timepoint=(row.get("timepoint") or "").strip() or None,
                threshold=th,
            )
            observations.append(obs.model_dump(mode="json"))

    if not observations:
        raise ValueError("CSV file contains no observation rows")
    return observations


def build_engine_input(
    csv_path: str | Path,
    *,
    query_id: str = "pilot_il6_01",
    scenario: str = "IL6_hypoxia_macrophage",
    target_entity: str = "IL6",
    target_level: MeasurementLevel = MeasurementLevel.MRNA,
    organism: str = "Homo sapiens",
    cell_type: str = "macrophage",
    control: str = "normoxia_vehicle",
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "scenario": scenario,
        "target": {"entity": target_entity, "level": target_level.value},
        "context": {
            "organism": organism,
            "cell_type": cell_type,
            "study_design": {"control": control},
        },
        "observations": csv_to_observations(csv_path),
    }


def csv_row_to_weighted_observation(row: dict[str, str | None]) -> WeightedObservation:
    """Converts a CSV row into a WeightedObservation model with strict field checks."""
    missing = [field for field in REQUIRED_WEIGHTED_COLUMNS if field not in row or row.get(field) is None]
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")

    raw_id = (row["observation_id"] or "").strip()
    raw_direction = (row["direction"] or "").strip()
    raw_confidence = (row["confidence"] or "").strip()
    raw_source = (row["source_type"] or "").strip()

    if not raw_id:
        raise ValueError("Empty observation_id is not allowed")
    if raw_direction not in ALLOWED_WEIGHTED_DIRECTIONS:
        raise ValueError(f"Invalid direction {raw_direction!r}; expected one of {sorted(ALLOWED_WEIGHTED_DIRECTIONS)}")
    if not raw_source:
        raise ValueError("Empty source_type is not allowed")

    confidence_val = _parse_float(raw_confidence, field_name="confidence", strict=True)
    assert confidence_val is not None
    if not (0.0 <= confidence_val <= 1.0):
        raise ValueError(f"Invalid confidence value out of bounds [0.0, 1.0]: {confidence_val}")

    return WeightedObservation(
        observation_id=raw_id,
        direction=raw_direction,  # type: ignore[arg-type]
        confidence=confidence_val,
        source_type=raw_source,
    )


def load_weighted_observations_from_csv(csv_path: str | Path) -> list[WeightedObservation]:
    """Loads WeightedObservation records from UTF-8/UTF-8-SIG CSV with duplicate ID check."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or missing header row")

        headers = {name.strip() for name in reader.fieldnames if name is not None}
        missing_cols = [col for col in REQUIRED_WEIGHTED_COLUMNS if col not in headers]
        if missing_cols:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing_cols)}")

        seen_ids: set[str] = set()
        items: list[WeightedObservation] = []
        for idx, row in enumerate(reader, start=1):
            if None in row and row[None]:
                raise ValueError(f"Row {idx} has extra unmapped columns: {row[None]}")
            obs = csv_row_to_weighted_observation(row)
            if obs.observation_id in seen_ids:
                raise ValueError(f"Duplicate observation_id detected: {obs.observation_id!r}")
            seen_ids.add(obs.observation_id)
            items.append(obs)

    return items


def evaluate_csv_observations(
    csv_path: str | Path,
    policy: ScoringPolicy | None = None,
) -> ScoringTrace:
    """Loads observations from CSV and computes weighted hypothesis score."""
    observations = load_weighted_observations_from_csv(csv_path)
    return evaluate_weighted_hypothesis(observations=observations, policy=policy)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/csv_to_observations.py <input.csv> [output.json]")

    in_csv = Path(sys.argv[1])
    payload = build_engine_input(in_csv)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)

    if len(sys.argv) >= 3:
        out_json = Path(sys.argv[2])
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(rendered, encoding="utf-8")
        print(f"Saved: {out_json} ({len(payload['observations'])} observations)")
    else:
        print(rendered)
