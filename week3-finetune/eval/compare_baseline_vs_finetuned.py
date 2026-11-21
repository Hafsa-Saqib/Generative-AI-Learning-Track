"""Compare the prompt-only baseline with the fine-tuned extractor on the same test split."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import (
    FINETUNED_8B_INPUT_USD_PER_MILLION,
    FINETUNED_8B_OUTPUT_USD_PER_MILLION,
    HAIKU_INPUT_USD_PER_MILLION,
    HAIKU_OUTPUT_USD_PER_MILLION,
    cost_per_thousand,
    field_accuracy,
    percentile,
)

def _latency(rows: list[dict]) -> dict:
    values = [row["latency_ms"] for row in rows if "latency_ms" in row]
    return {"p50": percentile(values, 50), "p95": percentile(values, 95)}


def _rates(model: str | None) -> tuple[float, float]:
    name = model or ""
    if "SmolLM" in name or name.startswith("local"):
        return 0.0, 0.0
    if "haiku" in name:
        return HAIKU_INPUT_USD_PER_MILLION, HAIKU_OUTPUT_USD_PER_MILLION
    return FINETUNED_8B_INPUT_USD_PER_MILLION, FINETUNED_8B_OUTPUT_USD_PER_MILLION


def compare(baseline: dict, finetuned: dict | None, tokens: dict) -> dict:
    input_rate, output_rate = _rates(baseline.get("model"))
    report = {
        "test_examples": len(baseline["rows"]),
        "baseline": {
            "model": baseline.get("model"),
            "accuracy": field_accuracy(baseline["rows"]),
            "latency_ms": baseline.get("latency_ms") or _latency(baseline["rows"]),
            "usd_per_1000_events": cost_per_thousand(tokens["input"], tokens["output"], input_rate, output_rate),
        },
    }
    if finetuned is None:
        report["finetuned"] = None
        report["decision"] = (
            "The fine-tuned model has not been scored. Keep the prompt-only baseline "
            "until a completed LoRA job beats it on accuracy, latency, and cost together."
        )
        return report
    tuned_accuracy = field_accuracy(finetuned["rows"])
    tuned_in, tuned_out = _rates(finetuned.get("model"))
    report["finetuned"] = {
        "model": finetuned.get("model"),
        "accuracy": tuned_accuracy,
        "latency_ms": finetuned.get("latency_ms") or _latency(finetuned["rows"]),
        "usd_per_1000_events": cost_per_thousand(tokens["input"], tokens["output"], tuned_in, tuned_out),
    }
    base_overall = report["baseline"]["accuracy"]["overall"]
    tuned_overall = tuned_accuracy["overall"]
    if baseline.get("parsed") == 0 and finetuned.get("parsed") == 0:
        report["decision"] = (
            f"On the held-out notes the adapter scores {tuned_overall:.1%} and the untuned model scores {base_overall:.1%}. "
            "Neither reply is a complete rubric JSON, so a person still reads the note. Contract facts stay in the library."
        )
    elif tuned_overall > base_overall:
        report["decision"] = (
            "Use the local adapter to read the carrier note. It is more accurate than the same model with no adapter."
        )
    else:
        report["decision"] = (
            "Keep the untuned model for reading the note. The adapter did not raise accuracy on the held-out test notes."
        )
    return report


def main() -> None:
    local_baseline = ROOT / "outputs" / "local_baseline_results.json"
    baseline_path = local_baseline if local_baseline.exists() else ROOT / "outputs" / "baseline_results.json"
    if not baseline_path.exists():
        raise SystemExit("Run eval/score_local.py first. There is no baseline number to beat yet.")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    local_tuned = ROOT / "outputs" / "local_finetuned_results.json"
    tuned_path = local_tuned if local_tuned.exists() else ROOT / "outputs" / "finetuned_results.json"
    finetuned = json.loads(tuned_path.read_text(encoding="utf-8")) if tuned_path.exists() else None
    sample = baseline["rows"][0]
    tokens = {"input": max(1, len(sample["note"].split()) * 2), "output": 40}
    report = compare(baseline, finetuned, tokens)
    target = ROOT / "outputs" / "comparison.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if "--fail-below-baseline" in __import__("sys").argv and finetuned is not None:
        if finetuned and report["finetuned"]["accuracy"]["overall"] < report["baseline"]["accuracy"]["overall"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
