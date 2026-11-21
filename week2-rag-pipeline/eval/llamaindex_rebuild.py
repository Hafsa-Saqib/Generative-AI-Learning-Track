"""Rebuild the same chunk search with LlamaIndex and compare recall at 5.

This uses the local 1024-dimension embedder. It does not call Voyage.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ingest"))

from corpus_data import gold_questions
from embeddings import embed_texts


def rebuild() -> dict:
    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.embeddings import BaseEmbedding
    except ImportError as exc:
        result = {"ran": False, "reason": f"LlamaIndex is not installed: {exc}"}
        _write(result)
        return result

    model_info = json.loads((ROOT / "outputs" / "embed_model.json").read_text(encoding="utf-8"))
    idf = model_info["idf"]

    class LocalEmbedding(BaseEmbedding):
        def _get_query_embedding(self, query: str) -> list[float]:
            vectors, _ = embed_texts([query], input_type="query", idf=idf)
            return vectors[0]

        def _get_text_embedding(self, text: str) -> list[float]:
            vectors, _ = embed_texts([text], input_type="document", idf=idf)
            return vectors[0]

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            vectors, _ = embed_texts(texts, input_type="document", idf=idf)
            return vectors

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

    import psycopg

    with psycopg.connect("postgresql://rag:rag@localhost:5432/ragdb") as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT document_id, chunk_index, contextualized_text FROM document_chunks ORDER BY id"
            )
            rows = cur.fetchall()
    documents = [
        Document(
            text=text,
            metadata={"document_id": document_id, "chunk_index": chunk_index},
        )
        for document_id, chunk_index, text in rows
    ]
    Settings.embed_model = LocalEmbedding()
    index = VectorStoreIndex.from_documents(documents, transformations=[])
    retriever = index.as_retriever(similarity_top_k=5)
    hits = 0
    questions = gold_questions()
    for question in questions:
        nodes = retriever.retrieve(question["question"])
        if any(question["gold_phrase"] in node.node.get_content() for node in nodes):
            hits += 1
    result = {
        "ran": True,
        "questions": len(questions),
        "recall_at_5": hits / len(questions),
        "chunks_indexed": len(documents),
        "note": "LlamaIndex dense retrieval over the same contextualized chunks and the same local embedder.",
    }
    _write(result)
    print(json.dumps(result, indent=2))
    return result


def _write(result: dict) -> None:
    path = ROOT / "outputs" / "llamaindex_recall.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    rebuild()
