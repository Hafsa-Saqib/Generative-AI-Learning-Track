"""Stop a retrieved chunk that carries instructions before it reaches generation."""

import re

INSTRUCTION = re.compile(
    r"ignore (?:all |any |prior |previous )?instructions|"
    r"disregard the contract|"
    r"always respond|"
    r"confirm a full refund|"
    r"confirm same-day|"
    r"you are now",
    re.I,
)


def guard_chunk(chunk: dict) -> dict:
    text = chunk.get("text") or chunk.get("contextualized_text") or ""
    matched = INSTRUCTION.search(text)
    flagged = matched is not None
    return {
        **chunk,
        "injection_flagged": flagged,
        "injection_excerpt": matched.group(0) if matched else None,
    }


def filter_chunks(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    checked = [guard_chunk(chunk) for chunk in chunks]
    kept = [chunk for chunk in checked if not chunk["injection_flagged"]]
    blocked = [chunk for chunk in checked if chunk["injection_flagged"]]
    return kept, blocked
