"""Streaming HTTP entry for one shipment question."""

import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.pipeline import handle
from app.routing import route_extraction, route_generation

app = FastAPI(title="Shipment generation")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/communications")
def communications(payload: dict):
    question = payload["question"]
    result = handle(question, payload.get("note", ""), payload.get("caller", "customer"))
    result["routing"] = {
        "extraction": route_extraction(),
        "generation": route_generation(question),
    }

    def chunks():
        yield json.dumps({"trace_id": result["trace_id"], "stage": "safety"}) + "\n"
        yield json.dumps(result) + "\n"

    return StreamingResponse(chunks(), media_type="application/x-ndjson")
