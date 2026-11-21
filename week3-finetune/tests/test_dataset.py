import json
from pathlib import Path

from schema import ExtractionResult, label_signals

ROOT = Path(__file__).resolve().parents[1]


def _rows(name):
    path = ROOT / "data" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_split_sizes_and_sacred_test():
    raw = _rows("raw_notes.jsonl")
    train = _rows("labeled_train.jsonl")
    val = _rows("labeled_val.jsonl")
    test = _rows("labeled_test.jsonl")
    assert len(raw) >= 300
    assert (len(train), len(val), len(test)) == (200, 50, 50)
    assert not ({row["id"] for row in test} & {row["id"] for row in train + val})


def test_null_coverage_and_schema():
    labeled = _rows("labeled_train.jsonl") + _rows("labeled_val.jsonl") + _rows("labeled_test.jsonl")
    nulls = [row for row in labeled if row["label"]["blocking_reason"] is None]
    assert len(nulls) / len(labeled) >= 0.10
    for row in labeled:
        ExtractionResult.model_validate(row["label"])
    carriers = {row["carrier"] for row in labeled}
    assert carriers == {"DHL", "FedEx", "UPS"}


def test_contradiction_rule_keeps_the_document_block():
    label = label_signals({
        "subtype": "customs_hold",
        "reason": "customs_documentation",
        "date": None,
        "contradiction": True,
        "commodity": "dry_goods",
    })
    assert label["blocking_reason"] == "customs_documentation"
    assert label["estimated_resolution_date"] is None


def test_two_dates_stay_null():
    label = label_signals({
        "subtype": "customs_hold",
        "reason": "customs_documentation",
        "date": "thu",
        "date_conflict": True,
        "commodity": "electronics",
    })
    assert label["estimated_resolution_date"] is None
