"""Turn the top 5 chunks into the Week 1 communication shape.

A commitment is copied only when that exact sentence was retrieved. With an
Anthropic key, the Week 1 Claude generator can replace this extractor.
"""

import re
import sys
from pathlib import Path

WEEK1 = Path(__file__).resolve().parents[2] / "week1-generation-pipeline"
if str(WEEK1) not in sys.path:
    sys.path.insert(0, str(WEEK1))

from pipeline.schema import ExceptionCommunication


def _cite(question: dict, chunks: list[dict]) -> dict | None:
    if not chunks:
        return None
    query = set(re.findall(r"[a-z0-9]+", question["question"].lower()))

    def score(chunk: dict) -> int:
        present = set(re.findall(r"[a-z0-9]+", chunk["raw_text"].lower()))
        return sum(1 for token in query if token in present)

    return max(chunks, key=score)


def answer(question: dict, chunks: list[dict]) -> dict:
    blob = "\n".join(chunk["contextualized_text"] for chunk in chunks)
    promise = question.get("promise")
    if promise and promise in blob:
        cited = next(chunk for chunk in chunks if promise in chunk["contextualized_text"])
        draft = {
            "exception_id": question["id"],
            "customer_facing_draft": promise,
            "internal_summary": "The commitment is a sentence retrieved from the library.",
            "grounded_commitments": [promise],
            "citations": [{
                "source_id": cited["document_id"],
                "clause_or_case": cited["raw_text"].splitlines()[0][:160],
            }],
            "confidence": 0.84,
            "requires_human_review": False,
            "review_reason": None,
        }
    else:
        cited = _cite(question, chunks)
        draft = {
            "exception_id": question["id"],
            "customer_facing_draft": "The timeline is under review.",
            "internal_summary": "No retrieved sentence supports a customer promise.",
            "grounded_commitments": [],
            "citations": [] if cited is None else [{
                "source_id": cited["document_id"],
                "clause_or_case": cited["raw_text"].splitlines()[0][:160],
            }],
            "confidence": 0.34,
            "requires_human_review": True,
            "review_reason": "No retrieved clause supports a customer promise for this request.",
        }
    ExceptionCommunication.model_validate(draft)
    return draft
