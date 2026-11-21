"""Dense HNSW search, sparse tsvector search, reciprocal rank fusion, then a top-5 rerank."""

import json
import os
import time
from pathlib import Path

import psycopg

from ingest.embeddings import embed_texts, tokens
from retrieve.rrf import reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[1]
DSN = "postgresql://rag:rag@localhost:5432/ragdb"
RRF_K = 60
COHERE_GAP_SECONDS = 7
_last_cohere_call = 0.0


def load_idf() -> tuple[dict, str]:
    payload = json.loads((ROOT / "outputs" / "embed_model.json").read_text(encoding="utf-8"))
    return payload["idf"], payload["model"]


def _rows(cursor) -> list[dict]:
    columns = [desc.name for desc in cursor.description]
    found = []
    for record in cursor.fetchall():
        row = dict(zip(columns, record))
        if isinstance(row.get("metadata"), str):
            row["metadata"] = json.loads(row["metadata"])
        found.append(row)
    return found


def dense_search(conn, vector: list[float]) -> list[dict]:
    literal = "[" + ",".join(f"{value:.7f}" for value in vector) + "]"
    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 40")
        cur.execute(
            """
            SELECT document_id, chunk_index, raw_text, contextualized_text, metadata::text AS metadata,
                   embedding <=> %s::vector AS distance
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 20
            """,
            (literal, literal),
        )
        return _rows(cur)


def sparse_search(conn, question: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH lexemes AS (
                SELECT string_agg(lexeme, ' | ') AS expr
                FROM unnest(to_tsvector('english', %s)) AS lex(lexeme, positions)
            )
            SELECT c.document_id, c.chunk_index, c.raw_text, c.contextualized_text,
                   c.metadata::text AS metadata,
                   ts_rank(c.search_text, to_tsquery('english', lexemes.expr)) AS rank
            FROM document_chunks c, lexemes
            WHERE lexemes.expr IS NOT NULL
              AND c.search_text @@ to_tsquery('english', lexemes.expr)
            ORDER BY rank DESC
            LIMIT 20
            """,
            (question,),
        )
        return _rows(cur)


def _cohere_rerank(question: str, candidates: list[dict]) -> list[dict]:
    """Trial keys allow 10 calls a minute, so each question waits its turn."""
    global _last_cohere_call
    import cohere
    from cohere.errors.too_many_requests_error import TooManyRequestsError

    client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    documents = [row["contextualized_text"] for row in candidates]
    for attempt in range(4):
        pause = COHERE_GAP_SECONDS - (time.monotonic() - _last_cohere_call)
        if _last_cohere_call and pause > 0:
            time.sleep(pause)
        try:
            response = client.rerank(
                model="rerank-v3.5",
                query=question,
                documents=documents,
                top_n=5,
            )
            _last_cohere_call = time.monotonic()
            return [candidates[item.index] for item in response.results]
        except TooManyRequestsError:
            if attempt == 3:
                raise
            print("Cohere trial limit is 10 calls a minute. Waiting 60 seconds.", flush=True)
            time.sleep(60)
            _last_cohere_call = time.monotonic()
    raise RuntimeError("Cohere rerank did not return a result.")


def rerank(question: str, candidates: list[dict], idf: dict) -> tuple[list[dict], str]:
    if os.environ.get("COHERE_API_KEY"):
        print("Cohere rerank", flush=True)
        return _cohere_rerank(question, candidates), "cohere-rerank-v3.5"
    query_tokens = set(tokens(question))

    def score(row: dict) -> float:
        present = set(tokens(row["contextualized_text"]))
        return sum(idf.get(token, 1.0) for token in query_tokens if token in present)

    ordered = sorted(candidates, key=score, reverse=True)
    return ordered[:5], "local-term-overlap"


def search(question: str, query_vector: list[float] | None = None, embedding_model: str | None = None) -> dict:
    idf, model = load_idf()
    if query_vector is None:
        vectors, used_model = embed_texts([question], input_type="query", idf=idf)
        query_vector = vectors[0]
    else:
        used_model = embedding_model or model
    with psycopg.connect(DSN) as conn:
        dense = dense_search(conn, query_vector)
        try:
            sparse = sparse_search(conn, question)
        except psycopg.Error:
            conn.rollback()
            sparse = []
    dense_keys = [(row["document_id"], row["chunk_index"]) for row in dense]
    sparse_keys = [(row["document_id"], row["chunk_index"]) for row in sparse]
    fused_keys = reciprocal_rank_fusion([dense_keys, sparse_keys], k=RRF_K)[:20]
    by_key = {(row["document_id"], row["chunk_index"]): row for row in dense + sparse}
    fused = [by_key[key] for key in fused_keys if key in by_key]
    top, reranker = rerank(question, fused, idf)
    return {
        "embedding_model": used_model or model,
        "reranker": reranker,
        "dense": len(dense),
        "sparse": len(sparse),
        "top": top,
    }
