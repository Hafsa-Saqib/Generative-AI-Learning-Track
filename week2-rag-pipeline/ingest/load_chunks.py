"""Chunk, situate, embed, and load document_chunks. Indexes come after the rows."""

import hashlib
import json
import os
import re
from pathlib import Path

import psycopg

from build_corpus import write_new_files
from chunkers import structure_aware_chunker
from corpus_data import document_type
from embeddings import embed_texts, fit_idf
from situate import situate

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
OUTPUTS = ROOT / "outputs"
DSN = "postgresql://rag:rag@localhost:5432/ragdb"

FOLDERS = ("sla_contracts", "policy_docs", "resolution_history")


def header_fields(text: str) -> dict:
    def grab(label: str) -> str:
        match = re.search(rf"(?m)^{label}: (.+)$", text)
        return match.group(1).strip() if match else ""

    title = ""
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return {
        "title": title,
        "carrier": grab("Carrier"),
        "country": grab("Country"),
        "commodity": grab("Commodity class"),
    }


def collect_rows() -> tuple[list[dict], dict, str]:
    pending = []
    for folder in FOLDERS:
        for path in sorted((CORPUS / folder).glob("*.md")):
            text = path.read_text(encoding="utf-8-sig")
            meta = header_fields(text)
            for chunk in structure_aware_chunker(text, max_size=512):
                note, writer = situate(chunk.text, path.name, meta)
                stored = note + "\n\n" + chunk.text
                pending.append({
                    "document_id": path.stem,
                    "document_type": document_type(folder),
                    "chunk_index": chunk.chunk_index,
                    "raw_text": chunk.text,
                    "contextualized_text": stored,
                    "content_hash": hashlib.sha256(stored.encode()).hexdigest(),
                    "metadata": {**meta, "source": path.name, "context_writer": writer},
                })
    idf = fit_idf([row["contextualized_text"] for row in pending])
    vectors, model = embed_texts(
        [row["contextualized_text"] for row in pending],
        input_type="document",
        idf=idf,
    )
    for row, vector in zip(pending, vectors):
        row["embedding"] = vector
        row["embedding_model_version"] = model
    return pending, idf, model


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.7f}" for value in values) + "]"


def load() -> dict:
    if not os.environ.get("VOYAGE_API_KEY"):
        raise SystemExit("Set VOYAGE_API_KEY in this terminal first. Loading without it would replace the Voyage vectors.")
    write_new_files()
    rows, idf, model = collect_rows()
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "embed_model.json").write_text(
        json.dumps({"model": model, "idf": idf}, indent=2),
        encoding="utf-8",
    )
    with psycopg.connect(DSN) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("TRUNCATE document_chunks RESTART IDENTITY")
        conn.execute("DROP INDEX IF EXISTS document_chunks_embedding_hnsw")
        conn.execute("DROP INDEX IF EXISTS document_chunks_search_gin")
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO document_chunks (
                    document_id, document_type, chunk_index, raw_text, contextualized_text,
                    embedding, content_hash, embedding_model_version, metadata
                ) VALUES (
                    %(document_id)s, %(document_type)s, %(chunk_index)s, %(raw_text)s,
                    %(contextualized_text)s, %(embedding)s::vector, %(content_hash)s,
                    %(embedding_model_version)s, %(metadata)s::jsonb
                )
                """,
                [
                    {
                        **row,
                        "embedding": vector_literal(row["embedding"]),
                        "metadata": json.dumps(row["metadata"]),
                    }
                    for row in rows
                ],
            )
        conn.execute(
            """
            ALTER TABLE document_chunks
            ADD COLUMN IF NOT EXISTS search_text tsvector
            GENERATED ALWAYS AS (to_tsvector('english', coalesce(contextualized_text, ''))) STORED
            """
        )
        conn.execute(
            """
            CREATE INDEX document_chunks_embedding_hnsw
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
        conn.execute(
            "CREATE INDEX document_chunks_search_gin ON document_chunks USING gin (search_text)"
        )
        conn.commit()
    summary = {
        "rows": len(rows),
        "documents": len({row["document_id"] for row in rows}),
        "model": model,
        "claude_notes": sum(row["metadata"]["context_writer"] == "claude" for row in rows),
        "template_notes": sum(row["metadata"]["context_writer"] == "template" for row in rows),
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    load()
