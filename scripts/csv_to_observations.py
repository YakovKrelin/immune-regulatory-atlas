from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from hypothesis_engine_v0_1.schemas import MeasurementLevel, Observation, SignalDirection, ThresholdSpec


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


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    val = raw.strip()
    if not val or val.lower() in {"na", "nan", "none", "null"}:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


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
        raise


def csv_to_observations(
    csv_path: str | Path,
    *,
    default_assay: str = "RNA-seq",
    default_level: MeasurementLevel = MeasurementLevel.MRNA,
    threshold: ThresholdSpec | None = None,
) -> list[dict[str, Any]]:
    th = threshold or ThresholdSpec()
    observations: list[dict[str, Any]] = []

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=1):
            entity = (row.get("entity") or "").strip()
            if not entity:
                continue

            log2fc = _parse_float(row.get("log2FC"))
            fdr = _parse_float(row.get("fdr"))
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