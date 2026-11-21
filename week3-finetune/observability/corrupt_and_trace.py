"""Inject one wrong SLA number and find it from the trace alone."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observability.instrumentation import find_bad_value, run_demo
BAD = "within 4 hours"


def chunks() -> list[dict]:
    return [
        {
            "id": "germany-4.2.2",
            "text": "4.2.2 DHL will send a status update and the next action within 4 hours of that notification.",
            "dense_score": 0.91,
            "sparse_score": 0.4,
            "rerank_score": 0.88,
        },
        {
            "id": "germany-4.2.3",
            "text": "4.2.3 This clause does not promise customs release, a delivery date, or a refund.",
            "dense_score": 0.55,
            "sparse_score": 0.2,
            "rerank_score": 0.40,
        },
        {
            "id": "brazil-5.2.2",
            "text": "This agreement does not promise customs release, a delivery date, or a refund.",
            "dense_score": 0.33,
            "sparse_score": 0.5,
            "rerank_score": 0.22,
        },
    ]


def main() -> None:
    query = "What can we promise after a 52 hour customs hold?"
    trace = run_demo(chunks(), query)
    found = find_bad_value(trace, BAD)
    target = ROOT / "outputs" / "bad_retrieval_trace.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps({"trace": json.loads(trace), "found_from_trace": found}, indent=2), encoding="utf-8")
    print(json.dumps(found))


if __name__ == "__main__":
    main()
