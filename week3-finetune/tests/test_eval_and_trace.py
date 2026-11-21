from eval.metrics import cost_per_thousand, field_accuracy, percentile, training_cost
from eval.score_local import parse_prediction
from finetune.format_for_fireworks import to_fireworks_format
from observability.instrumentation import find_bad_value, run_demo


def test_field_accuracy_and_cost_math():
    rows = [{
        "label": {
            "exception_subtype": "customs_hold",
            "blocking_reason": "customs_documentation",
            "estimated_resolution_date": "2026-11-05",
            "requires_cold_chain_monitoring": True,
        },
        "prediction": {
            "exception_subtype": "customs_hold",
            "blocking_reason": "customs_inspection",
            "estimated_resolution_date": "2026-11-05",
            "requires_cold_chain_monitoring": True,
        },
    }]
    scores = field_accuracy(rows)
    assert scores["exception_subtype"] == 1
    assert scores["blocking_reason"] == 0
    assert scores["overall"] == 0.75
    assert percentile([10, 20, 30, 40], 50) == 30
    assert cost_per_thousand(1000, 1000, 1, 5) == 6
    budget = training_cost(200_000, 3)
    assert budget["billed_tokens"] == 600_000
    assert budget["usd"] == 0.3


def test_fireworks_messages_use_the_label_as_the_assistant_turn():
    rows = to_fireworks_format([{
        "note": "DHL: held — missing commercial invoice.",
        "label": {"exception_subtype": "customs_hold", "blocking_reason": "customs_documentation"},
    }])
    roles = [turn["role"] for turn in rows[0]["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "customs_documentation" in rows[0]["messages"][2]["content"]


def test_a_valid_json_reply_is_kept_and_a_broken_reply_is_a_miss():
    parsed, ok = parse_prediction(
        'Note:\n{"exception_subtype":"customs_hold","blocking_reason":"customs_documentation",'
        '"estimated_resolution_date":null,"requires_cold_chain_monitoring":false}'
    )
    assert ok
    assert parsed["blocking_reason"] == "customs_documentation"
    missed, parsed_ok = parse_prediction("I cannot tell.")
    assert parsed_ok is False
    assert missed["exception_subtype"] == "unparsed"
    partial, complete = parse_prediction(
        '{"exception_subtype":"broken_seal","blocking_reason":"seal_broken",'
        '"estimated_resolution_date":null,"requires_cold_chain_monitoring":true}'
    )
    assert complete is False
    assert partial["exception_subtype"] == "unparsed"
    assert partial["blocking_reason"] == "seal_broken"
    assert partial["requires_cold_chain_monitoring"] is True


def test_trace_alone_finds_the_bad_sla_number():
    chunks = [
        {"id": "good", "text": "status update within 48 hours", "dense_score": 0.2, "sparse_score": 0.2, "rerank_score": 0.2},
        {"id": "bad", "text": "status update within 4 hours", "dense_score": 0.9, "sparse_score": 0.1, "rerank_score": 0.4},
    ]
    found = find_bad_value(run_demo(chunks, "48 hour hold"), "within 4 hours")
    assert found["stage"] == "retrieval.dense"
    assert found["trace_id"]
