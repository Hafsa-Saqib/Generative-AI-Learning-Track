import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    chunk_index: int
    metadata: dict


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def fixed_size_chunker(text: str, size: int = 512, overlap: int = 50) -> list[Chunk]:
    tokens = ENC.encode(text)
    chunks = []
    start = 0
    index = 0
    step = size - overlap
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(Chunk(ENC.decode(tokens[start:end]), index, {}))
        if end == len(tokens):
            break
        start += step
        index += 1
    return chunks


def structure_aware_chunker(markdown_text: str, max_size: int = 512) -> list[Chunk]:
    sections = []
    current = []
    for line in markdown_text.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    clauses = []
    for section in sections:
        parts = re.split(r"(?m)(?=^\d+\.\d+(?:\.\d+)*\s)", section)
        for part in parts:
            part = part.strip()
            if part:
                clauses.append(part)

    pieces = []
    for clause in clauses:
        if count_tokens(clause) <= max_size:
            pieces.append(clause)
        else:
            pieces.extend(chunk.text for chunk in fixed_size_chunker(clause, size=max_size, overlap=20))

    return [Chunk(text, index, {}) for index, text in enumerate(pieces)]


def clause_was_split(chunks: list[Chunk]) -> bool:
    for chunk in chunks:
        has_start = "exceeds 48 hours" in chunk.text
        has_end = "within 48 hours of that notification" in chunk.text
        if has_start and not has_end:
            return True
    return False


if __name__ == "__main__":
    document = Path("corpus/sla_contracts/dhl_germany_cold_chain.md").read_text(encoding="utf-8")
    demo_size = 80

    print("FIXED SIZE")
    fixed = fixed_size_chunker(document, size=demo_size, overlap=10)
    for chunk in fixed:
        print(f"--- piece {chunk.chunk_index} ({count_tokens(chunk.text)} tokens)")
        print(chunk.text)
        print()
    print("4.2.2 split:", clause_was_split(fixed))

    print("STRUCTURE AWARE")
    structured = structure_aware_chunker(document, max_size=demo_size)
    for chunk in structured:
        print(f"--- piece {chunk.chunk_index} ({count_tokens(chunk.text)} tokens)")
        print(chunk.text)
        print()
    print("4.2.2 split:", clause_was_split(structured))