"""
report_generator.py

Turns a list of EvalResult objects into:
  - a per-transcript + aggregate Markdown report (human review)
  - a raw JSON dump (machine-readable, for dashboards/regression tracking)
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .evaluator import EvalResult
from .rubric import Criterion, DEFAULT_RUBRIC


def _score_bar(score: float, max_score: int = 5, width: int = 10) -> str:
    filled = round((score / max_score) * width)
    return "█" * filled + "░" * (width - filled)


def _aggregate_stats(results: list[EvalResult], rubric: list[Criterion]) -> dict:
    overall_scores = [r.overall_score for r in results if r.overall_score > 0]
    per_criterion: dict[str, list[int]] = {c.key: [] for c in rubric}
    all_flags: list[str] = []
    failures = 0

    for r in results:
        if "judge_output_unparseable" in r.flags:
            failures += 1
        all_flags.extend(r.flags)
        for cr in r.criterion_results:
            if cr.score is not None:
                per_criterion[cr.key].append(cr.score)

    flag_counts: dict[str, int] = {}
    for f in all_flags:
        flag_counts[f] = flag_counts.get(f, 0) + 1

    return {
        "n_transcripts": len(results),
        "n_failed_to_parse": failures,
        "mean_overall": round(statistics.mean(overall_scores), 2) if overall_scores else 0.0,
        "median_overall": round(statistics.median(overall_scores), 2) if overall_scores else 0.0,
        "stdev_overall": round(statistics.stdev(overall_scores), 2) if len(overall_scores) > 1 else 0.0,
        "per_criterion_mean": {
            k: round(statistics.mean(v), 2) if v else None for k, v in per_criterion.items()
        },
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda kv: -kv[1])),
    }


def generate_markdown_report(
    results: list[EvalResult],
    rubric: list[Criterion] = DEFAULT_RUBRIC,
    title: str = "Voice AI Transcript Evaluation Report",
) -> str:
    stats = _aggregate_stats(results, rubric)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [f"# {title}", "", f"_Generated {now}_", ""]

    # --- Summary section ---
    lines += [
        "## Summary",
        "",
        f"- Transcripts evaluated: **{stats['n_transcripts']}**",
        f"- Failed to parse (judge output unusable): **{stats['n_failed_to_parse']}**",
        f"- Mean overall score: **{stats['mean_overall']} / 5** {_score_bar(stats['mean_overall'])}",
        f"- Median overall score: **{stats['median_overall']} / 5**",
        f"- Std dev: **{stats['stdev_overall']}**",
        "",
        "### Mean score by criterion",
        "",
        "| Criterion | Mean Score | |",
        "|---|---|---|",
    ]
    for c in rubric:
        mean = stats["per_criterion_mean"].get(c.key)
        if mean is None:
            lines.append(f"| {c.name} | n/a | |")
        else:
            lines.append(f"| {c.name} | {mean} / 5 | {_score_bar(mean)} |")
    lines.append("")

    if stats["flag_counts"]:
        lines += ["### Flags raised", ""]
        for flag, count in stats["flag_counts"].items():
            lines.append(f"- `{flag}` — {count} transcript(s)")
        lines.append("")

    # --- Per-transcript detail ---
    lines += ["## Per-Transcript Results", ""]
    for r in results:
        lines.append(f"### {r.transcript_id}")
        lines.append("")
        lines.append(f"- **Overall score:** {r.overall_score} / 5 (judge: `{r.judge_name}`)")
        if r.flags:
            lines.append(f"- **Flags:** {', '.join(f'`{f}`' for f in r.flags)}")
        if r.transcript_warnings:
            lines.append(f"- **Parsing warnings:** {len(r.transcript_warnings)} (see appendix)")
        if r.parse_errors:
            lines.append(f"- ⚠️ **Judge response issues:** {len(r.parse_errors)} (see appendix)")
        lines.append("")
        lines.append("| Criterion | Score | Rationale |")
        lines.append("|---|---|---|")
        for cr in r.criterion_results:
            score_display = f"{cr.score}/5" if cr.score is not None else "—"
            rationale = (cr.rationale or "").replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {cr.key} | {score_display} | {rationale} |")
        lines.append("")
        if r.summary:
            lines.append(f"**Summary:** {r.summary}")
            lines.append("")
        lines.append("---")
        lines.append("")

    # --- Appendix: warnings/errors, for debugging the pipeline itself ---
    any_warnings = any(r.transcript_warnings or r.parse_errors for r in results)
    if any_warnings:
        lines += ["## Appendix: Parsing & Judge-Output Warnings", ""]
        for r in results:
            if not (r.transcript_warnings or r.parse_errors):
                continue
            lines.append(f"### {r.transcript_id}")
            for w in r.transcript_warnings:
                lines.append(f"- [transcript parsing] {w}")
            for e in r.parse_errors:
                lines.append(f"- [judge output] {e}")
            lines.append("")

    return "\n".join(lines)


def generate_json_report(results: list[EvalResult], rubric: list[Criterion] = DEFAULT_RUBRIC) -> str:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": _aggregate_stats(results, rubric),
        "results": [r.to_dict() for r in results],
    }
    return json.dumps(payload, indent=2)


def write_reports(results: list[EvalResult], out_dir: str | Path, rubric: list[Criterion] = DEFAULT_RUBRIC) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "eval_report.md"
    json_path = out_dir / "eval_report.json"
    md_path.write_text(generate_markdown_report(results, rubric), encoding="utf-8")
    json_path.write_text(generate_json_report(results, rubric), encoding="utf-8")
    return md_path, json_path
