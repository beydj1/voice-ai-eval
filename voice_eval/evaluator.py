"""
evaluator.py

Orchestrates the LLM-as-judge evaluation:
  1. Build a strict JSON-only prompt from the rubric + transcript
  2. Call the (swappable) judge provider
  3. Parse + validate the JSON response (retry once on malformed output)
  4. Compute the weighted overall score
  5. Return a structured EvalResult, ready for reporting or aggregation
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

from .judge_providers import JudgeProvider
from .rubric import Criterion, DEFAULT_RUBRIC, rubric_as_prompt_block, weighted_overall
from .transcript_loader import Transcript

SYSTEM_PROMPT = """You are a strict, consistent QA evaluator for voice AI (voice assistant / IVR / voicebot) \
call transcripts. You will be given a rubric with several criteria and a transcript. \
Score EACH criterion independently on a 1-5 integer scale using the scoring guide provided. \
Do not let a low score on one criterion drag down your judgment of another (e.g., an agent can \
be factually accurate but still conversationally awkward).

Base every score only on evidence in the transcript. If information needed to judge a criterion \
(e.g. whether a claim was factually correct) is not verifiable from the transcript alone, say so \
in the rationale and score conservatively (3) rather than guessing.

Respond with ONLY a single JSON object, no markdown fences, no prose outside the JSON, in exactly \
this shape:

{
  "scores": {
    "<criterion_key>": {"score": <int 1-5>, "rationale": "<1-3 sentences, cite specific turns>"},
    ...
  },
  "summary": "<2-4 sentence overall assessment of the call>",
  "flags": ["<short tag>", ...]   // e.g. ["escalation_needed", "pii_exposure", "hallucinated_policy"] or [] if none
}"""

USER_PROMPT_TEMPLATE = """## Rubric

{rubric_block}

## Transcript to evaluate (id: {transcript_id})

{dialogue}

## Instructions

Score this transcript against every criterion above. Return only the JSON object."""


@dataclass
class CriterionResult:
    key: str
    score: int | None
    rationale: str


@dataclass
class EvalResult:
    transcript_id: str
    judge_name: str
    criterion_results: list[CriterionResult]
    overall_score: float
    summary: str
    flags: list[str]
    parse_errors: list[str] = field(default_factory=list)
    transcript_warnings: list[str] = field(default_factory=list)
    raw_judge_output: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _extract_json(text: str) -> dict:
    """Judges sometimes wrap JSON in markdown fences or add stray text.
    Try direct parse first, then fall back to extracting the outermost {...}.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))  # let this raise if it still fails

    raise ValueError("No JSON object found in judge output.")


def evaluate_transcript(
    transcript: Transcript,
    provider: JudgeProvider,
    rubric: list[Criterion] = DEFAULT_RUBRIC,
    max_retries: int = 1,
) -> EvalResult:
    """Run one transcript through one judge provider and return a structured result."""
    rubric_block = rubric_as_prompt_block(rubric)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        rubric_block=rubric_block,
        transcript_id=transcript.id,
        dialogue=transcript.as_dialogue_string() or "[EMPTY TRANSCRIPT -- no turns parsed]",
    )

    parse_errors: list[str] = []
    parsed: dict | None = None
    raw_output = ""

    for attempt in range(max_retries + 1):
        raw_output = provider.complete(SYSTEM_PROMPT, user_prompt)
        try:
            parsed = _extract_json(raw_output)
            break
        except (ValueError, json.JSONDecodeError) as e:
            parse_errors.append(f"Attempt {attempt + 1}: {e}")
            continue

    if parsed is None:
        return EvalResult(
            transcript_id=transcript.id,
            judge_name=provider.name,
            criterion_results=[CriterionResult(key=c.key, score=None, rationale="No score: judge output unparseable.") for c in rubric],
            overall_score=0.0,
            summary="EVALUATION FAILED: judge did not return parseable JSON after retries.",
            flags=["judge_output_unparseable"],
            parse_errors=parse_errors,
            transcript_warnings=transcript.warnings,
            raw_judge_output=raw_output,
        )

    raw_scores = parsed.get("scores", {})
    criterion_results = []
    numeric_scores: dict[str, float] = {}
    for c in rubric:
        entry = raw_scores.get(c.key)
        if not entry or "score" not in entry:
            criterion_results.append(CriterionResult(key=c.key, score=None, rationale="Missing from judge response."))
            parse_errors.append(f"Judge response missing criterion '{c.key}'.")
            continue
        try:
            score = int(entry["score"])
        except (TypeError, ValueError):
            score = None
            parse_errors.append(f"Non-integer score for '{c.key}': {entry.get('score')!r}")
        if score is not None and not (1 <= score <= 5):
            parse_errors.append(f"Score out of range [1-5] for '{c.key}': {score}")
        criterion_results.append(CriterionResult(key=c.key, score=score, rationale=entry.get("rationale", "")))
        if score is not None:
            numeric_scores[c.key] = score

    overall = weighted_overall(numeric_scores, rubric)

    return EvalResult(
        transcript_id=transcript.id,
        judge_name=provider.name,
        criterion_results=criterion_results,
        overall_score=overall,
        summary=parsed.get("summary", ""),
        flags=parsed.get("flags", []),
        parse_errors=parse_errors,
        transcript_warnings=transcript.warnings,
        raw_judge_output=raw_output,
    )


def evaluate_batch(
    transcripts: list[Transcript],
    provider: JudgeProvider,
    rubric: list[Criterion] = DEFAULT_RUBRIC,
) -> list[EvalResult]:
    return [evaluate_transcript(t, provider, rubric) for t in transcripts]
