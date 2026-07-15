# Voice AI Transcript Evaluator (LLM-as-Judge)

A self-contained pipeline that scores voice AI call transcripts against a
multi-criteria rubric using an LLM judge. Built to handle messy, mixed-format
real-world transcript exports without needing them pre-normalized first.

## What it does

1. **Loads transcripts** from plain text, JSON, or CSV — auto-detected per
   file, so you can point it at a folder with mixed formats. Unknown/missing
   fields degrade gracefully with warnings instead of crashing.
2. **Judges each transcript** against a 4-criterion rubric:
   - Task Completion / Goal Success (35%)
   - Conversational Quality — tone, naturalness, empathy (25%)
   - Accuracy / Correctness of Information (25%)
   - Safety / Compliance (15%)

   (Weights and criteria are trivially editable in `voice_eval/rubric.py`.)
3. **Computes a weighted overall score** (1–5) per transcript plus aggregate
   stats (mean, median, stdev, per-criterion breakdown, flag counts) across
   a batch.
4. **Generates reports**: a human-readable Markdown report and a
   machine-readable JSON dump (for dashboards / regression tracking over
   time — e.g. comparing agent versions).

## Why it's swappable

The judge is an abstract `JudgeProvider` with one method: `complete(system, user) -> str`.
Three are included out of the box:

| Provider | Use case |
|---|---|
| `AnthropicJudge` | Claude via the Anthropic SDK |
| `OpenAIJudge` | GPT via the OpenAI SDK |
| `MockJudge` | No API calls — deterministic fake scores, for testing the pipeline itself |

There's also `CallableJudge`, which wraps *any* `function(system, user) -> str`
— point it at an internal gateway, a local model server, or a second vendor's
SDK without touching the rest of the codebase. This mirrors the same
multi-LLM pattern used in a RAG-based eval setup: same rubric, same report
format, swap only the judge.

## Quick start

```bash
# 1. Test the full pipeline with zero API calls / zero setup:
python cli.py --input examples/ --provider mock --out outputs/

# 2. Real run with Claude as judge:
export ANTHROPIC_API_KEY=sk-ant-...
pip install anthropic
python cli.py --input path/to/your/transcripts/ --provider anthropic --model claude-sonnet-4-6 --out outputs/

# 3. Real run with GPT as judge:
export OPENAI_API_KEY=sk-...
pip install openai
python cli.py --input path/to/your/transcripts/ --provider openai --model gpt-4o --out outputs/
```

Output: `outputs/eval_report.md` and `outputs/eval_report.json`.

## Supported transcript formats

The loader auto-detects format per file (by extension first, then content
sniffing). It's forgiving about field names — any of these work:

**Plain text** — `Speaker: text` or `Speaker (timestamp): text` per line:
```
Agent (00:00): Thanks for calling, how can I help?
User (00:04): My dishwasher won't drain.
```

**JSON** — a `turns`/`messages`/`transcript`/`dialogue` list, or a bare list.
Speaker key can be `speaker`/`role`/`from`/`author`; text key can be
`text`/`content`/`message`/`utterance`:
```json
{
  "turns": [
    {"role": "agent", "text": "Thanks for calling...", "timestamp": "0:00"},
    {"role": "user", "text": "My dishwasher won't drain.", "timestamp": "0:04"}
  ]
}
```

**CSV** — header row with `speaker`/`role` + `text`/`content` columns
(falls back to first two columns if headers don't match):
```csv
timestamp,speaker,text
0:00,bot,Thanks for calling...
0:04,caller,My dishwasher won't drain.
```

Speaker labels are normalized (`caller`/`customer`/`human` → `user`;
`bot`/`assistant`/`ivr` → `agent`) so the rubric prompt is consistent
regardless of how your source system labels turns.

## Using it as a library

```python
from voice_eval import load_transcript, get_provider, evaluate_transcript, generate_markdown_report

t = load_transcript("call_001.json")
judge = get_provider("anthropic", model="claude-sonnet-4-6")
result = evaluate_transcript(t, judge)

print(result.overall_score)      # 4.25
print(result.flags)              # e.g. ["escalation_needed"]
for cr in result.criterion_results:
    print(cr.key, cr.score, cr.rationale)
```

## Editing the rubric

Edit `voice_eval/rubric.py`. Each `Criterion` has a `key`, `weight`,
`description`, and `scoring_guide` (the 1-5 anchors shown to the judge).
Add, remove, or reweight criteria there — nothing else needs to change,
since the prompt, parser, and report all read from the same rubric list.

## Reliability notes

- The judge is prompted for **strict JSON only**; the parser also handles
  markdown-fenced JSON or stray prose around it, and retries once on
  unparseable output before marking a transcript as failed
  (`judge_output_unparseable` flag) rather than silently guessing scores.
- Per-transcript parsing warnings (e.g. "line before any recognized speaker
  label") and judge-response issues (e.g. missing criterion, out-of-range
  score) are both surfaced in the report's Appendix — useful for catching
  transcript export weirdness or judge drift before it corrupts your metrics.

## Project layout

```
voice_eval/
  transcript_loader.py   # multi-format parsing → normalized Transcript
  rubric.py               # criteria, weights, scoring guides
  judge_providers.py      # Anthropic / OpenAI / Mock / Callable judges
  evaluator.py             # prompt building, JSON parsing, scoring
  report_generator.py     # Markdown + JSON report output
cli.py                     # command-line entrypoint
examples/                  # one transcript per supported format
```
