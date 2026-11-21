"""Held-out questions: recall, context precision, and faithfulness.

recall_at_5 is the share of the 30 questions whose gold sentence appears in the
top 5 chunks. context_precision_at_5 is the average, across those questions, of
(chunks in the top 5 that contain the gold sentence) / 5. A single answering
chunk therefore scores 0.20 on that question. Both numbers are written to
outputs/search_results.json under metrics.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ingest"))

from corpus_data import gold_questions
from embeddings import embed_texts
from retrieve.answer import answer
from retrieve.search import load_idf, search


def average_precision(chunks: list[dict], phrase: str) -> float:
    hits = 0
    total = 0.0
    for rank, chunk in enumerate(chunks, start=1):
        if phrase in chunk["contextualized_text"]:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


def score_question(question: dict, query_vector: list[float], embedding_model: str) -> dict:
    found = search(question["question"], query_vector=query_vector, embedding_model=embedding_model)
    top = found["top"]
    blob = "\n".join(chunk["contextualized_text"] for chunk in top)
    relevant = [chunk for chunk in top if question["gold_phrase"] in chunk["contextualized_text"]]
    draft = answer(question, top)
    claims = draft["grounded_commitments"]
    faithful = all(claim in blob for claim in claims) if claims else True
    return {
        "id": question["id"],
        "question": question["question"],
        "gold_document": question["gold_document"],
        "gold_phrase": question["gold_phrase"],
        "hit": bool(relevant),
        "context_precision_at_5": len(relevant) / 5,
        "average_precision": average_precision(top, question["gold_phrase"]),
        "decision_match": draft["requires_human_review"] == question["expects_human_review"],
        "faithful": faithful,
        "draft": draft,
        "top": [
            {
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "relevant": question["gold_phrase"] in chunk["contextualized_text"],
                "text": chunk["raw_text"],
            }
            for chunk in top
        ],
        "embedding_model": found["embedding_model"],
        "reranker": found["reranker"],
    }


def run() -> dict:
    questions = gold_questions()
    idf, _stored_model = load_idf()
    print(f"Embedding {len(questions)} questions in one Voyage call", flush=True)
    vectors, model = embed_texts([question["question"] for question in questions], input_type="query", idf=idf)
    details = [
        score_question(question, vector, model)
        for question, vector in zip(questions, vectors)
    ]
    count = len(details) or 1
    metrics = {
        "questions": len(details),
        "recall_at_5": sum(item["hit"] for item in details) / count,
        "context_precision_at_5": sum(item["context_precision_at_5"] for item in details) / count,
        "average_precision": sum(item["average_precision"] for item in details) / count,
        "faithfulness": sum(item["faithful"] for item in details) / count,
        "decision_match": sum(item["decision_match"] for item in details) / count,
        "embedding_model": details[0]["embedding_model"] if details else "",
        "reranker": details[0]["reranker"] if details else "",
        "dense_index": "hnsw m=16 ef_construction=64 cosine",
        "sparse_index": "gin tsvector english",
        "rrf_k": 60,
    }
    payload = {"metrics": metrics, "questions": details}
    out = ROOT / "outputs" / "search_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    misses = [item["id"] for item in details if not item["hit"]]
    print("misses:", ", ".join(misses) if misses else "none")
    return payload


if __name__ == "__main__":
    run()
