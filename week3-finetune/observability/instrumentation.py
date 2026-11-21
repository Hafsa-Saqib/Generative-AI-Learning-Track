"""Spans for embed, retrieve, extract, and generate. One trace id covers the whole request."""

import json
import uuid
from contextlib import contextmanager


class MemoryTracer:
    def __init__(self):
        self.traces = []

    @contextmanager
    def span(self, name: str, trace_id: str, attributes: dict):
        record = {"trace_id": trace_id, "name": name, "attributes": dict(attributes)}
        self.traces.append(record)
        try:
            yield record
        except Exception as exc:
            record["attributes"]["error"] = type(exc).__name__
            raise

    def dump(self) -> str:
        return json.dumps(self.traces, indent=2)


def traced_stage(tracer: MemoryTracer, trace_id: str, stage_name: str, query: str, results: list[dict]) -> list[dict]:
    with tracer.span(
        f"retrieval.{stage_name}",
        trace_id,
        {
            "gen_ai.operation.name": "retrieval",
            "retrieval.query": query,
            "retrieval.result_count": len(results),
            "retrieval.result_ids": [row["id"] for row in results],
            "retrieval.scores": [row.get("score", 0) for row in results],
            "retrieval.texts": [row["text"] for row in results],
        },
    ):
        return results


def find_bad_value(trace_json: str, bad_value: str) -> dict:
    """Read only the trace. Report the first stage whose retrieved text contains the bad value."""
    spans = json.loads(trace_json)
    for span in spans:
        texts = span.get("attributes", {}).get("retrieval.texts", [])
        if any(bad_value in text for text in texts):
            return {"stage": span["name"], "trace_id": span["trace_id"]}
    return {"stage": None, "trace_id": spans[0]["trace_id"] if spans else None}


def run_demo(chunks: list[dict], query: str) -> str:
    tracer = MemoryTracer()
    trace_id = uuid.uuid4().hex
    with tracer.span("embed.query", trace_id, {"gen_ai.operation.name": "embeddings", "retrieval.query": query}):
        pass
    dense = sorted(chunks, key=lambda row: row["dense_score"], reverse=True)
    sparse = sorted(chunks, key=lambda row: row["sparse_score"], reverse=True)
    traced_stage(tracer, trace_id, "dense", query, dense)
    traced_stage(tracer, trace_id, "sparse", query, sparse)
    fused = sorted(chunks, key=lambda row: row["dense_score"] + row["sparse_score"], reverse=True)
    traced_stage(tracer, trace_id, "fusion", query, fused)
    reranked = sorted(chunks, key=lambda row: row["rerank_score"], reverse=True)[:5]
    traced_stage(tracer, trace_id, "rerank", query, reranked)
    with tracer.span("generation", trace_id, {"gen_ai.operation.name": "chat", "retrieval.query": query}):
        pass
    return tracer.dump()
