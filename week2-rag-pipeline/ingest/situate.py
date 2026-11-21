"""Situating notes stored in front of each chunk.

Saved Claude notes live in outputs/claude_notes.json. The Germany 4.2.2 note
from the first live call is the fallback for that one clause. Any chunk
without a saved note gets a short template.
"""

import hashlib
import json
from pathlib import Path

NOTES_PATH = Path(__file__).resolve().parents[1] / "outputs" / "claude_notes.json"
_NOTES: dict | None = None

CLAUDE_422 = (
    "This chunk is from Section 4.2 (\"Customs-hold response\") of the DHL Germany "
    "cold-chain service agreement, which applies specifically to cold-chain shipments "
    "to Germany. It details DHL's commitment to provide a status update and next steps "
    "within 48 hours if an HE customs hold exceeds 48 hours after customer notification. "
    "This is clarified by the following clause (4.2.3), which states this does not "
    "guarantee customs release, delivery date, or refund—distinguishing it from the "
    "broader exclusions listed in Section 4.3 regarding electronics shipments to Brazil "
    "and same-day customs release promises."
)


def chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def saved_notes() -> dict:
    global _NOTES
    if _NOTES is None:
        if NOTES_PATH.exists():
            _NOTES = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
        else:
            _NOTES = {}
    return _NOTES


def situate(chunk_text: str, source_name: str, meta: dict) -> tuple[str, str]:
    saved = saved_notes().get(source_name, {}).get(chunk_id(chunk_text))
    if saved:
        return saved["note"], saved.get("writer", "claude")
    if source_name == "dhl_germany_cold_chain.md" and chunk_text.startswith("4.2.2"):
        return CLAUDE_422, "claude"
    title = meta.get("title") or source_name
    first = chunk_text.splitlines()[0]
    note = (
        f"This chunk is from {title}. "
        f"Carrier {meta.get('carrier') or 'unknown'}, "
        f"country {meta.get('country') or 'unknown'}, "
        f"commodity {meta.get('commodity') or 'unknown'}. "
        f"The chunk begins with: {first}"
    )
    return note, "template"
