"""Count tokens in the long warehouse appendix and measure Claude prompt cache when a key is set."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ingest"))

from chunkers import count_tokens, structure_aware_chunker

DOCUMENT = ROOT / "corpus" / "sla_contracts" / "dhl_germany_cold_chain_expanded.md"


def main() -> dict:
    text = DOCUMENT.read_text(encoding="utf-8-sig")
    chunks = structure_aware_chunker(text, max_size=512)
    result = {
        "document": DOCUMENT.name,
        "document_tokens": count_tokens(text),
        "chunks": len(chunks),
        "cache_write_tokens": None,
        "cache_read_tokens": None,
        "note": "",
    }
    if not os.environ.get("ANTHROPIC_API_KEY"):
        result["note"] = (
            "ANTHROPIC_API_KEY is not set in this session, so the long document was not sent. "
            "The earlier live call on the short Germany contract was 410 input tokens, cache write 0, cache read 0."
        )
    else:
        from anthropic import Anthropic

        client = Anthropic()
        usages = []
        for chunk in chunks[:2]:
            response = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<document>\n{text}\n</document>",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": f"Give a 50 to 100 token situating note for this chunk:\n{chunk.text}",
                        },
                    ],
                }],
            )
            usages.append(response.usage)
        result["cache_write_tokens"] = getattr(usages[0], "cache_creation_input_tokens", 0)
        result["cache_read_tokens"] = getattr(usages[-1], "cache_read_input_tokens", 0) if len(usages) > 1 else 0
        result["note"] = "Two chunks of the long document were sent with the same cached prefix."
    out = ROOT / "outputs" / "cache_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
