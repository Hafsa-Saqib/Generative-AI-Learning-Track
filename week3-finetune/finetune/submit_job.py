"""Train the local LoRA adapter and record the plan.

This replaces a paid Fireworks job. The base model stays frozen, the adapter
is a few megabytes, and the training cost on this machine is zero.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_plan(formatted_rows: list[dict], tokens: int) -> dict:
    from finetune.train_local import BASE_MODEL, EPOCHS, MAX_STEPS, RANK

    return {
        "submitted": False,
        "host": "local",
        "base_model": BASE_MODEL,
        "lora_rank": RANK,
        "epochs": EPOCHS,
        "max_steps": MAX_STEPS,
        "rows": len(formatted_rows),
        "training_tokens": tokens,
        "billed_tokens": tokens * EPOCHS,
        "usd": 0.0,
        "rate_usd_per_million": 0.0,
        "source": "Local CPU LoRA with Hugging Face PEFT. No paid fine-tune host.",
        "reason": (
            "The base model is frozen and only the low-rank matrices are trained on this PC. "
            "Training cost is $0. A paid host such as Fireworks is not used."
        ),
    }


def main() -> None:
    from eval.metrics import load_jsonl
    from finetune.format_for_fireworks import to_fireworks_format
    from finetune.train_local import train

    rows = to_fireworks_format(load_jsonl(ROOT / "data" / "labeled_train.jsonl"))
    out = ROOT / "outputs" / "train.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    plan = build_plan(rows, training_token_count(rows))
    summary = train(rows)
    plan["submitted"] = True
    plan["job_id"] = "local-lora"
    plan.update({key: summary[key] for key in ("trainable_parameters", "total_parameters", "trainable_fraction", "adapter_dir")})
    target = ROOT / "outputs" / "finetune_plan.json"
    target.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


def training_token_count(rows: list[dict]) -> int:
    week2 = ROOT.parent / "week2-rag-pipeline" / "ingest"
    sys.path.insert(0, str(week2))
    from chunkers import count_tokens

    total = 0
    for row in rows:
        for turn in row["messages"]:
            total += count_tokens(turn["content"])
    return total


if __name__ == "__main__":
    main()
