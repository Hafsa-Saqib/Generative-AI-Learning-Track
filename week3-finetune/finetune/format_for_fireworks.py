"""Turn labeled notes into Fireworks supervised-fine-tune messages."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = (
    "Extract exception_subtype, blocking_reason, estimated_resolution_date, "
    "and requires_cold_chain_monitoring from the carrier note. "
    "Reply with one JSON object. Use null when the note does not support a value."
)


def to_fireworks_format(labeled_examples: list[dict]) -> list[dict]:
    """Fireworks SFT rows are chat messages. The assistant turn is the target JSON."""
    formatted = []
    for example in labeled_examples:
        formatted.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": example["note"]},
                {"role": "assistant", "content": json.dumps(example["label"], separators=(",", ":"))},
            ]
        })
    return formatted


def main() -> None:
    source = ROOT / "data" / "labeled_train.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    formatted = to_fireworks_format(rows)
    target = ROOT / "outputs" / "fireworks_train.jsonl"
    target.parent.mkdir(exist_ok=True)
    target.write_text("".join(json.dumps(row) + "\n" for row in formatted), encoding="utf-8")
    print(json.dumps({"rows": len(formatted), "path": str(target)}))


if __name__ == "__main__":
    main()
