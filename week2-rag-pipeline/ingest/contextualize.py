import os
from pathlib import Path

from anthropic import Anthropic

from chunkers import structure_aware_chunker

client = Anthropic()
MODEL = "claude-sonnet-5"

CONTEXT_PROMPT = """<document>
{full_document}
</document>
Here is a chunk from this document:
<chunk>
{chunk_text}
</chunk>
Give a short context, about 50 to 100 tokens, explaining what this chunk is about and how it relates to the rest of the document. This context will be prepended to the chunk to improve search. Answer only with the context."""


def contextualize_chunk(full_document: str, chunk_text: str) -> tuple[str, object]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<document>\n{full_document}\n</document>",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": CONTEXT_PROMPT.format(full_document="", chunk_text=chunk_text),
                },
            ],
        }],
    )
    context = response.content[0].text.strip()
    return context, response.usage


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in this terminal first.")

    document = Path("corpus/sla_contracts/dhl_germany_cold_chain.md").read_text(encoding="utf-8")
    chunks = structure_aware_chunker(document, max_size=80)
    clause = next(chunk for chunk in chunks if chunk.text.startswith("4.2.2"))
    context, usage = contextualize_chunk(document, clause.text)

    print("CONTEXT")
    print(context)
    print("---")
    print("STORED TEXT")
    print(context + "\n\n" + clause.text)
    print("---")
    print("input tokens:", usage.input_tokens)
    print("cache write tokens:", getattr(usage, "cache_creation_input_tokens", 0))
    print("cache read tokens:", getattr(usage, "cache_read_input_tokens", 0))