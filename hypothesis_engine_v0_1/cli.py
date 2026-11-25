from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hypothesis_engine_v0_1.engine import run_hypothesis_engine
from hypothesis_engine_v0_1.ollama_client import OllamaClientConfig, OllamaExplanationClient
from hypothesis_engine_v0_1.postprocessor import compute_canonical_sha256
from hypothesis_engine_v0_1.renderer import RenderReportRequest, render_hypothesis_report
from hypothesis_engine_v0_1.schemas import HypothesisEngineInput


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypothesis-engine-cli",
        description="Run BIOOS Hypothesis Engine v0.1 from a JSON input file.",
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to a JSON file matching HypothesisEngineInput.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print structured JSON output instead of Markdown.",
    )
    parser.add_argument(
        "--max-bullets",
        type=int,
        default=5,
        help="Maximum bullets per section in Markdown output (1-20).",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Optional local Ollama model name for LLM rendering (uses deterministic renderer if omitted).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL when --ollama-model is set.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not exit with status 1 when post-processor reports violations.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.input_json.exists():
        print(f"Error: input file not found: {args.input_json}", file=sys.stderr)
        return 2

    raw_data = json.loads(args.input_json.read_text(encoding="utf-8-sig"))
    engine_input = HypothesisEngineInput.model_validate(raw_data)
    engine_output = run_hypothesis_engine(engine_input)

    llm_client = (
        OllamaExplanationClient(
            OllamaClientConfig(
                base_url=args.ollama_url,
                model=args.ollama_model,
            )
        )
        if args.ollama_model
        else None
    )

    render_res = render_hypothesis_report(
        RenderReportRequest(
            engine_input=engine_input,
            engine_output=engine_output,
            max_bullets_per_section=args.max_bullets,
        ),
        llm_client=llm_client,
    )

    input_hash = compute_canonical_sha256(engine_input)
    output_hash = compute_canonical_sha256(engine_output)

    if args.as_json:
        payload = {
            "engine_output": engine_output.model_dump(mode="json"),
            "markdown": render_res.markdown,
            "used_llm": render_res.used_llm,
            "validation_passed": render_res.validation_passed,
            "violations": render_res.violations,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "explanation_hash": render_res.explanation_hash,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render_res.markdown)
        if render_res.violations:
            print("\n### Post-Processor Violations", file=sys.stderr)
            for violation in render_res.violations:
                print(f"- {violation}", file=sys.stderr)

    if not args.no_strict and not render_res.validation_passed:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

