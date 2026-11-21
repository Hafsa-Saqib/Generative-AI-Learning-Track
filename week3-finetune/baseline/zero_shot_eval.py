"""Claude Haiku zero-shot baseline for the untouched test split.

This is the number a fine-tune has to beat. The script does not invent a score
when the Anthropic key is missing.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schema import ExtractionResult
TEST_PATH = ROOT / "data" / "labeled_test.jsonl"
OUT_PATH = ROOT / "outputs" / "baseline_results.json"
MODEL = "claude-haiku-4-5"

SYSTEM = """Extract four fields from one carrier note. Return only JSON with keys exception_subtype, blocking_reason, estimated_resolution_date, requires_cold_chain_monitoring.
exception_subtype is one of customs_hold, weather_delay, delivery_exception, pickup_missed, damage, address_exception, breakdown, dangerous_goods_check, seal_issue, unknown.
blocking_reason is one of customs_documentation, customs_inspection, weather, receiver_closed, shipper_dock_closed, address_incomplete, damage, mechanical_breakdown, dangerous_goods, seal_broken, or null.
A missing document is customs_documentation. An exam is customs_inspection. A closed receiver is not a closed shipper dock.
estimated_resolution_date uses the week of Monday 2026-11-02: Tuesday 2026-11-03, Wednesday 2026-11-04, Thursday 2026-11-05, Friday 2026-11-06. Use null when the note gives two days, says maybe, or gives no day.
requires_cold_chain_monitoring is true for cold-chain or pharma freight and for a temperature problem. A dry shipment that says temp nominal is false.
If the note has no exception, use unknown, null, null, false. Do not guess."""


def zero_shot_extract(note_text: str, client) -> ExtractionResult:
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM,
        messages=[{"role": "user", "content": note_text}],
    )
    parts = [block.text for block in response.content if hasattr(block, "text")]
    text = "\n".join(parts).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return ExtractionResult.model_validate(json.loads(text))


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in this terminal first. No baseline score is invented without it.")
    from anthropic import Anthropic

    from eval.metrics import field_accuracy

    client = Anthropic()
    rows = []
    latencies = []
    for line in TEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        started = time.perf_counter()
        prediction = zero_shot_extract(item["note"], client)
        latencies.append((time.perf_counter() - started) * 1000)
        rows.append({**item, "prediction": prediction.model_dump()})
        time.sleep(0.3)
    OUT_PATH.parent.mkdir(exist_ok=True)
    payload = {
        "model": MODEL,
        "rows": rows,
        "accuracy": field_accuracy(rows),
        "latency_ms": {"p50": sorted(latencies)[len(latencies) // 2], "p95": sorted(latencies)[max(0, round(0.95 * (len(latencies) - 1)))]},
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"model": MODEL, "accuracy": payload["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
