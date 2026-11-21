"""Offline gate for the scores already saved by Week 2 and the Week 3 dataset rules."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEEK2 = ROOT.parent / "week2-rag-pipeline"
sys.path.insert(0, str(ROOT))

THRESHOLDS = {
    "recall_at_5": 1.0,
    "faithfulness": 1.0,
    "decision_match": 1.0,
}


def check_week2() -> list[str]:
    path = WEEK2 / "outputs" / "search_results.json"
    if not path.exists():
        return ["Week 2 search_results.json is missing."]
    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    problems = []
    for name, floor in THRESHOLDS.items():
        if metrics.get(name, 0) < floor:
            problems.append(f"{name} {metrics.get(name)} is below {floor}")
    return problems


def check_dataset() -> list[str]:
    from eval.metrics import load_jsonl

    train = {row["id"] for row in load_jsonl(ROOT / "data" / "labeled_train.jsonl")}
    val = {row["id"] for row in load_jsonl(ROOT / "data" / "labeled_val.jsonl")}
    test = {row["id"] for row in load_jsonl(ROOT / "data" / "labeled_test.jsonl")}
    problems = []
    if train & test or val & test:
        problems.append("Held-out test ids appear in train or val.")
    if len(test) != 50 or len(val) != 50 or len(train) != 200:
        problems.append("Split sizes are not 200, 50, and 50.")
    return problems


def main() -> None:
    problems = check_week2() + check_dataset()
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)
    print(json.dumps({"week2": "pass", "dataset": "pass"}))


if __name__ == "__main__":
    main()
