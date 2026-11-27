from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from hypothesis_engine_v0_1.scoring_rules import (
    ScoringPolicy,
    WeightedObservation,
    evaluate_weighted_hypothesis,
)


class CliScoringInput(BaseModel):
    observations: list[WeightedObservation] = Field(default_factory=list)
    policy: ScoringPolicy | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate weighted observations from a JSON file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file containing observations and optional policy.",
    )
    return parser.parse_args(argv)


def run_cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)

    try:
        raw_text = input_path.read_text(encoding="utf-8")
        payload_data = json.loads(raw_text)
        cli_input = CliScoringInput.model_validate(payload_data)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"Error loading scoring input: {exc}", file=sys.stderr)
        return 1

    trace = evaluate_weighted_hypothesis(
        observations=cli_input.observations,
        policy=cli_input.policy,
    )
    print(trace.model_dump_json(indent=2))
    return 0


def main() -> None:
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
