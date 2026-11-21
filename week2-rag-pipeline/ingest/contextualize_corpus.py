"""Ask Claude for a situating note for every library chunk.

The full document is marked for prompt caching. Chunks are sent four at a time.
Progress is saved after each group, so a stopped run continues where it left off.
The original Germany 4.2.2 note is kept and is not requested again.
"""

import json
import os
import re
import time
from pathlib import Path

from anthropic import Anthropic, RateLimitError

from chunkers import structure_aware_chunker
from situate import CLAUDE_422, NOTES_PATH, chunk_id

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
FOLDERS = ("sla_contracts", "policy_docs", "resolution_history")
MODEL = "claude-sonnet-5"
GROUP_SIZE = 4

PROMPT = """The document above is the only source.
Write one situating note for each chunk below. Each note should be about 50 to 100 tokens.
Say which document, carrier, country, and commodity the chunk belongs to, and how it relates to the nearby clauses.
Do not add a promise that the document does not make.
Return a JSON array of strings, one string per chunk, in the same order. No other text.

Chunks:
{chunks}
"""


def load_store() -> dict:
    if NOTES_PATH.exists():
        return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    return {}


def save_store(store: dict) -> None:
    NOTES_PATH.parent.mkdir(exist_ok=True)
    NOTES_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def parse_notes(text: str, expected: int) -> list[str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    if not isinstance(data, list) or len(data) != expected:
        raise ValueError(f"Expected {expected} notes, got {type(data).__name__} of length {getattr(data, '__len__', lambda: '?')()}")
    return [str(item).strip() for item in data]


def message_text(response) -> str:
    """Sonnet 5 can lead with a thinking block. The notes are in the text block."""
    parts = [block.text for block in response.content if hasattr(block, "text")]
    if not parts:
        raise ValueError("Claude returned no text block.")
    return "\n".join(parts)


def request_notes(client: Anthropic, document: str, chunks: list[str]) -> list[str]:
    numbered = "\n\n".join(f"[{index}] {chunk}" for index, chunk in enumerate(chunks, start=1))
    for attempt in range(4):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<document>\n{document}\n</document>",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": PROMPT.format(chunks=numbered)},
                    ],
                }],
            )
            return parse_notes(message_text(response), len(chunks))
        except (RateLimitError, json.JSONDecodeError, ValueError) as exc:
            if attempt == 3:
                raise
            print(f"Retrying after {type(exc).__name__}. Waiting 30 seconds.", flush=True)
            time.sleep(30)
    raise RuntimeError("Claude did not return notes.")


def documents() -> list[Path]:
    paths = []
    for folder in FOLDERS:
        paths.extend(sorted((CORPUS / folder).glob("*.md")))
    return paths


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in this terminal first.")
    client = Anthropic()
    store = load_store()
    paths = documents()
    for path_index, path in enumerate(paths, start=1):
        document = path.read_text(encoding="utf-8-sig")
        chunks = structure_aware_chunker(document, max_size=512)
        file_notes = store.setdefault(path.name, {})
        pending = []
        for chunk in chunks:
            key = chunk_id(chunk.text)
            if path.name == "dhl_germany_cold_chain.md" and chunk.text.startswith("4.2.2"):
                file_notes[key] = {"note": CLAUDE_422, "writer": "claude"}
                continue
            if key in file_notes:
                continue
            pending.append(chunk)
        print(f"{path_index}/{len(paths)} {path.name}: {len(pending)} chunks left", flush=True)
        for start in range(0, len(pending), GROUP_SIZE):
            group = pending[start:start + GROUP_SIZE]
            notes = request_notes(client, document, [chunk.text for chunk in group])
            for chunk, note in zip(group, notes):
                file_notes[chunk_id(chunk.text)] = {"note": note, "writer": "claude"}
            save_store(store)
            print(f"  saved {len(group)} notes", flush=True)
            time.sleep(1)
    covered = sum(len(notes) for notes in store.values())
    print(json.dumps({"files": len(store), "notes": covered}))


if __name__ == "__main__":
    main()
