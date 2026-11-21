"""The 30 scored questions stay out of the prompt examples and the library text."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEEK1 = ROOT.parent / "week1-generation-pipeline"
sys.path.insert(0, str(ROOT / "ingest"))

from corpus_data import gold_questions


def test_thirty_questions_do_not_appear_in_prompt_examples_or_the_corpus():
    questions = gold_questions()
    assert len(questions) == 30
    assert len({item["id"] for item in questions}) == 30
    bank = json.loads((WEEK1 / "fixtures" / "few_shot_bank.json").read_text(encoding="utf-8"))
    prompt_examples = "\n".join(json.dumps(example) for example in bank)
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "corpus").rglob("*.md"))
    for item in questions:
        assert item["question"] not in prompt_examples
        assert item["question"] not in corpus


def test_saved_recall_and_context_precision_match_the_per_question_rows():
    payload = json.loads((ROOT / "outputs" / "search_results.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    rows = payload["questions"]
    assert metrics["questions"] == 30
    assert "recall_at_5" in metrics
    assert "context_precision_at_5" in metrics
    assert metrics["recall_at_5"] == sum(bool(row["hit"]) for row in rows) / len(rows)
    assert metrics["context_precision_at_5"] == pytest.approx(
        sum(row["context_precision_at_5"] for row in rows) / len(rows)
    )
    assert metrics["recall_at_5"] == 1.0
