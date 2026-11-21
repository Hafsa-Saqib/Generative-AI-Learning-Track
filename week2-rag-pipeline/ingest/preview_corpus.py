from pathlib import Path

from chunkers import structure_aware_chunker

TYPES = {
    "sla_contracts": "sla_contract",
    "policy_docs": "policy_doc",
    "resolution_history": "resolution_history",
}

root = Path("corpus")
for folder_name, document_type in TYPES.items():
    for path in sorted((root / folder_name).glob("*.md")):
        chunks = structure_aware_chunker(path.read_text(encoding="utf-8-sig"), max_size=80)
        print("=" * 60)
        print(f"{document_type} | {path.name} | {len(chunks)} chunks")
        for chunk in chunks:
            first = chunk.text.splitlines()[0]
            print(f"  {chunk.chunk_index}: {first}")
