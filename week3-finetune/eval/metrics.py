"""Field-level accuracy, latency, and cost for the extraction comparison."""

import json
from pathlib import Path

FIELDS = (
    "exception_subtype",
    "blocking_reason",
    "estimated_resolution_date",
    "requires_cold_chain_monitoring",
)

# Course rate card in section 3.6 and the Haiku-class baseline the build asks for.
# Re-verify both pages before spending money on a real job.
LORA_SFT_USD_PER_MILLION = 0.50
HAIKU_INPUT_USD_PER_MILLION = 1.00
HAIKU_OUTPUT_USD_PER_MILLION = 5.00
FINETUNED_8B_INPUT_USD_PER_MILLION = 0.20
FINETUNED_8B_OUTPUT_USD_PER_MILLION = 0.20


def field_accuracy(rows: list[dict]) -> dict:
    totals = {field: 0 for field in FIELDS}
    correct = {field: 0 for field in FIELDS}
    for row in rows:
        gold = row["label"]
        pred = row["prediction"]
        for field in FIELDS:
            totals[field] += 1
            if pred.get(field) == gold.get(field):
                correct[field] += 1
    per_field = {field: correct[field] / totals[field] for field in FIELDS}
    per_field["overall"] = sum(correct.values()) / sum(totals.values())
    return per_field


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def cost_per_thousand(input_tokens: int, output_tokens: int, input_rate: float, output_rate: float) -> float:
    one = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
    return one * 1000


def training_cost(training_tokens: int, epochs: int = 3) -> dict:
    billed = training_tokens * epochs
    return {
        "training_tokens": training_tokens,
        "epochs": epochs,
        "billed_tokens": billed,
        "usd": round(billed / 1_000_000 * LORA_SFT_USD_PER_MILLION, 4),
        "rate_usd_per_million": LORA_SFT_USD_PER_MILLION,
        "source": "Course section 3.6 Fireworks LoRA SFT rate. Re-verify before a paid job.",
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
