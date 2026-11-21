import json
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError

from pipeline.example_selector import WordCountEmbedder, select_examples
from pipeline.recover import human_review_fallback, inspect_payload, parse_model_json, unsupported_commitments
from pipeline.schema import ExceptionCommunication

ROOT = Path(__file__).resolve().parents[1]
BANK = json.loads((ROOT / "fixtures" / "few_shot_bank.json").read_text(encoding="utf-8"))


def _draft(**overrides) -> dict:
    payload = {
        "exception_id": "exc_001",
        "customer_facing_draft": "The timeline is under review.",
        "internal_summary": "No supported promise.",
        "grounded_commitments": [],
        "citations": [],
        "confidence": 0.4,
        "requires_human_review": True,
        "review_reason": "No clause supports a promise.",
    }
    payload.update(overrides)
    return payload


def test_template_renders_the_selected_examples_and_only_those():
    query = "HE | DHL | Germany | cold_chain | 52 hours | missing commercial invoice"
    embedder = WordCountEmbedder([example["input"] for example in BANK])
    chosen = select_examples(query, BANK, embedder.embed_texts, k=3)
    assert len(chosen) == 3
    assert all(example in BANK for example in chosen)
    prompt = Environment(loader=FileSystemLoader(ROOT / "templates")).get_template(
        "communication_prompt.jinja2"
    ).render(
        current_event={
            "exception_id": "exc_001",
            "status_code": "HE",
            "carrier": "DHL",
            "country": "Germany",
            "commodity": "cold_chain",
            "hours_in_status": 52,
            "freeform_note": "missing commercial invoice",
        },
        sla_snippets=[{"source_id": "sla_dhl_he_cold_de", "text": "DHL will send a status update."}],
        resolution_history=[{"summary": "A past hold was not treated as a release promise."}],
        few_shot_examples=chosen,
    )
    assert "sla_dhl_he_cold_de" in prompt
    assert chosen[0]["input"] in prompt
    assert all(example["input"] in prompt for example in chosen)
    left_out = next(example for example in BANK if example not in chosen)
    assert left_out["input"] not in prompt


def test_groq_fenced_json_parses_and_broken_json_does_not():
    raw = """```json
{"exception_id":"exc_008","customer_facing_draft":"The timeline is under review.","internal_summary":"markup","grounded_commitments":[],"citations":[],"confidence":0.2,"requires_human_review":true,"review_reason":"The reply was not safe to send."}
```"""
    draft = parse_model_json(raw)
    assert draft.exception_id == "exc_008"
    assert draft.requires_human_review is True
    with pytest.raises(ValidationError):
        parse_model_json("this is not json")


def test_claude_tool_payload_with_bad_shape_or_markup_is_rejected():
    broken, problems = inspect_payload({"exception_id": "exc_008"}, set())
    assert broken is None
    assert problems
    marked = _draft(
        customer_facing_draft="The timeline is under review. <parameter name=\"x\">",
        requires_human_review=True,
        review_reason="Tool markup.",
    )
    draft, problems = inspect_payload(marked, set())
    assert isinstance(draft, ExceptionCommunication)
    assert any("tool markup" in problem for problem in problems)


def test_a_promise_with_no_supporting_clause_goes_to_a_person():
    clause = "DHL will send a status update and the next action within 48 hours of that notification."
    assert unsupported_commitments(["We will refund the freight today."], clause) == ["We will refund the freight today."]
    assert unsupported_commitments([clause], clause) == []
    fallback = human_review_fallback("exc_011", "not json")
    assert fallback.requires_human_review is True
    assert fallback.grounded_commitments == []
    assert fallback.customer_facing_draft == "The timeline is under review."
