"""
transcript_loader.py

Loads voice AI transcripts from mixed/unknown formats (plain text, JSON, CSV)
and normalizes them into a common internal representation:

    Transcript = {
        "id": str,
        "metadata": dict,          # anything extra found (call_id, duration, agent_version, etc.)
        "turns": [
            {
                "speaker": str,     # normalized to "user" / "agent" / "system" / "<raw label>"
                "text": str,
                "timestamp": Optional[str],
                "raw": dict         # original row/object, kept for debugging
            },
            ...
        ]
    }

Design goal: never crash on "mixed" or messy real-world exports. Best-effort
parsing with a `warnings` list attached so the eval report can flag rows that
needed guessing.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SPEAKER_ALIASES = {
    "user": "user", "caller": "user", "customer": "user", "human": "user",
    "client": "user", "speaker_0": "user", "speaker0": "user", "user_1": "user",

    "agent": "agent", "assistant": "agent", "bot": "agent", "ai": "agent",
    "voice_agent": "agent", "system_ai": "agent", "speaker_1": "agent",
    "speaker1": "agent", "ivr": "agent",

    "system": "system", "operator": "system",
}

# e.g. "User: hello" / "AGENT (00:03): hi there" / "[Caller] hey"
LINE_PATTERN = re.compile(
    r"^\s*[\[\(]?\s*(?P<speaker>[A-Za-z][A-Za-z0-9 _\-]{0,30}?)\s*[\]\)]?"
    r"\s*(?:\((?P<ts>[\d:.,\s]+)\))?\s*[:\-]\s*(?P<text>.+)$"
)


@dataclass
class Turn:
    speaker: str
    text: str
    timestamp: str | None = None
    raw: Any = None


@dataclass
class Transcript:
    id: str
    turns: list[Turn]
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dialogue_string(self) -> str:
        """Render turns back to a clean, judge-friendly dialogue block."""
        lines = []
        for t in self.turns:
            ts = f" ({t.timestamp})" if t.timestamp else ""
            lines.append(f"{t.speaker.upper()}{ts}: {t.text}")
        return "\n".join(lines)


def _normalize_speaker(raw_label: str) -> str:
    key = raw_label.strip().lower().replace(" ", "_")
    return SPEAKER_ALIASES.get(key, raw_label.strip())


def _sniff_format(text_or_bytes: str, filename: str | None = None) -> str:
    if filename:
        ext = Path(filename).suffix.lower()
        if ext == ".json":
            return "json"
        if ext == ".csv":
            return "csv"
        if ext in (".txt", ".log", ".md"):
            return "text"

    stripped = text_or_bytes.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return "json"
        except json.JSONDecodeError:
            pass

    first_line = stripped.splitlines()[0] if stripped else ""
    if "," in first_line and ("speaker" in first_line.lower() or "role" in first_line.lower() or "text" in first_line.lower()):
        return "csv"

    return "text"


def _parse_json(content: str, warnings: list[str]) -> list[Turn]:
    data = json.loads(content)

    # Common shapes: {"turns": [...]}, {"messages": [...]}, or a bare list
    if isinstance(data, dict):
        turns_raw = data.get("turns") or data.get("messages") or data.get("transcript") or data.get("dialogue")
        if turns_raw is None:
            warnings.append("JSON object had no recognizable turns/messages/transcript key; treating as empty.")
            turns_raw = []
    elif isinstance(data, list):
        turns_raw = data
    else:
        warnings.append("Unrecognized JSON root type; no turns extracted.")
        turns_raw = []

    turns = []
    for item in turns_raw:
        if not isinstance(item, dict):
            warnings.append(f"Skipped non-object turn entry: {item!r}")
            continue
        speaker_raw = (
            item.get("speaker") or item.get("role") or item.get("from")
            or item.get("author") or "unknown"
        )
        text = item.get("text") or item.get("content") or item.get("message") or item.get("utterance") or ""
        if not text:
            warnings.append(f"Turn missing text field, kept as empty string: {item!r}")
        ts = item.get("timestamp") or item.get("time") or item.get("start_time")
        turns.append(Turn(speaker=_normalize_speaker(str(speaker_raw)), text=str(text), timestamp=str(ts) if ts else None, raw=item))
    return turns


def _parse_csv(content: str, warnings: list[str]) -> list[Turn]:
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        warnings.append("CSV had no header row; could not parse.")
        return []

    fieldnames_lower = {f.lower(): f for f in reader.fieldnames}
    speaker_col = next((fieldnames_lower[k] for k in ("speaker", "role", "from") if k in fieldnames_lower), None)
    text_col = next((fieldnames_lower[k] for k in ("text", "content", "message", "utterance") if k in fieldnames_lower), None)
    ts_col = next((fieldnames_lower[k] for k in ("timestamp", "time", "start_time") if k in fieldnames_lower), None)

    if speaker_col is None or text_col is None:
        warnings.append(
            f"CSV missing speaker/text columns (found: {reader.fieldnames}); "
            "falling back to first two columns as speaker,text."
        )

    turns = []
    for row in reader:
        if speaker_col and text_col:
            speaker_raw = row.get(speaker_col, "unknown")
            text = row.get(text_col, "")
        else:
            values = list(row.values())
            speaker_raw = values[0] if len(values) > 0 else "unknown"
            text = values[1] if len(values) > 1 else ""
        ts = row.get(ts_col) if ts_col else None
        turns.append(Turn(speaker=_normalize_speaker(str(speaker_raw)), text=str(text), timestamp=ts, raw=row))
    return turns


def _parse_text(content: str, warnings: list[str]) -> list[Turn]:
    turns = []
    current_speaker = None
    current_ts = None
    buffer_lines: list[str] = []

    def flush():
        if current_speaker is not None and buffer_lines:
            turns.append(Turn(speaker=_normalize_speaker(current_speaker), text=" ".join(buffer_lines).strip(), timestamp=current_ts, raw="\n".join(buffer_lines)))

    for line in content.splitlines():
        if not line.strip():
            continue
        m = LINE_PATTERN.match(line)
        if m:
            flush()
            current_speaker = m.group("speaker")
            current_ts = m.group("ts").strip() if m.group("ts") else None
            buffer_lines = [m.group("text").strip()]
        else:
            if current_speaker is None:
                warnings.append(f"Line before any recognized speaker label, attributing to 'unknown': {line!r}")
                current_speaker = "unknown"
                buffer_lines = [line.strip()]
            else:
                # Continuation of the previous turn (wrapped line)
                buffer_lines.append(line.strip())
    flush()

    if not turns:
        warnings.append("No 'Speaker: text' pattern detected anywhere in file.")
    return turns


def load_transcript(path: str | Path, transcript_id: str | None = None) -> Transcript:
    """Load a single transcript file, auto-detecting its format."""
    path = Path(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    return parse_transcript(content, filename=path.name, transcript_id=transcript_id or path.stem)


def parse_transcript(content: str, filename: str | None = None, transcript_id: str | None = None) -> Transcript:
    """Parse raw transcript content (any supported format) into a Transcript object."""
    warnings: list[str] = []
    fmt = _sniff_format(content, filename)

    metadata: dict = {"detected_format": fmt}
    if fmt == "json":
        turns = _parse_json(content, warnings)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                metadata.update({k: v for k, v in parsed.items() if k not in ("turns", "messages", "transcript", "dialogue")})
        except json.JSONDecodeError:
            warnings.append("Detected as JSON by sniffing but failed to parse; falling back to text parser.")
            turns = _parse_text(content, warnings)
            metadata["detected_format"] = "text (json fallback)"
    elif fmt == "csv":
        turns = _parse_csv(content, warnings)
    else:
        turns = _parse_text(content, warnings)

    empty_turns = sum(1 for t in turns if not t.text.strip())
    if empty_turns:
        warnings.append(f"{empty_turns} turn(s) had empty text after parsing.")

    return Transcript(
        id=transcript_id or "transcript",
        turns=turns,
        metadata=metadata,
        warnings=warnings,
    )


def load_transcripts_from_dir(dir_path: str | Path) -> list[Transcript]:
    """Load every supported file in a directory (mixed formats OK)."""
    dir_path = Path(dir_path)
    results = []
    for f in sorted(dir_path.iterdir()):
        if f.suffix.lower() in (".json", ".csv", ".txt", ".log", ".md"):
            results.append(load_transcript(f))
    return results
