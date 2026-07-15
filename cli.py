#!/usr/bin/env python3
"""
cli.py — run the LLM-as-judge voice AI evaluator from the command line.

Examples
--------
# Quick pipeline test, no API key needed:
python cli.py --input examples/ --provider mock --out outputs/

# Real run with Claude as judge:
python cli.py --input path/to/transcripts/ --provider anthropic --model claude-sonnet-4-6 --out outputs/

# Real run with GPT as judge:
python cli.py --input path/to/transcripts/ --provider openai --model gpt-4o --out outputs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voice_eval import (
    load_transcripts_from_dir,
    load_transcript,
    get_provider,
    evaluate_batch,
    write_reports,
)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluator for voice AI transcripts.")
    parser.add_argument("--input", required=True, help="Path to a transcript file OR a directory of mixed-format transcripts.")
    parser.add_argument("--provider", default="mock", choices=["anthropic", "openai", "mock"], help="Judge backend (default: mock, no API calls).")
    parser.add_argument("--model", default=None, help="Model name override, e.g. claude-sonnet-4-6 or gpt-4o.")
    parser.add_argument("--out", default="outputs", help="Directory to write eval_report.md / eval_report.json into.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        transcripts = load_transcripts_from_dir(input_path)
    elif input_path.is_file():
        transcripts = [load_transcript(input_path)]
    else:
        print(f"Error: {input_path} not found.", file=sys.stderr)
        sys.exit(1)

    if not transcripts:
        print(f"No supported transcript files found in {input_path} (expected .json/.csv/.txt/.log/.md).", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(transcripts)} transcript(s) from {input_path}.")
    for t in transcripts:
        fmt = t.metadata.get("detected_format", "?")
        n_warn = len(t.warnings)
        print(f"  - {t.id}: {len(t.turns)} turns, format={fmt}" + (f", {n_warn} parsing warning(s)" if n_warn else ""))

    provider_kwargs = {}
    if args.model:
        provider_kwargs["model"] = args.model
    provider = get_provider(args.provider, **provider_kwargs)

    print(f"\nRunning judge: {provider.name} ...")
    results = evaluate_batch(transcripts, provider)

    md_path, json_path = write_reports(results, args.out)
    print(f"\nWrote report:\n  {md_path}\n  {json_path}")

    failed = sum(1 for r in results if "judge_output_unparseable" in r.flags)
    if failed:
        print(f"\nWARNING: {failed}/{len(results)} transcript(s) failed to produce parseable judge output.", file=sys.stderr)


if __name__ == "__main__":
    main()
