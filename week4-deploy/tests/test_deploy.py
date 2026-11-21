from app.caching import SemanticCache
from app.index_manager import IndexAlias, chunks_to_embed, content_hash
from app.pipeline import handle
from app.routing import fallback_after_error, route_extraction, route_generation


def test_only_the_changed_document_is_reembedded():
    previous = {"a": content_hash("alpha"), "b": content_hash("beta")}
    documents = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta changed"}]
    assert chunks_to_embed(previous, documents) == ["b"]


def test_alias_swaps_only_when_scores_hold_and_rolls_back():
    alias = IndexAlias()
    alias.build("docs_index_v1", ["a"], {"recall_at_5": 1.0})
    assert alias.swap("docs_index_v1", {"recall_at_5": 1.0})
    alias.build("docs_index_v2", ["a", "b"], {"recall_at_5": 0.5})
    assert alias.swap("docs_index_v2", {"recall_at_5": 1.0}) is False
    assert alias.current == "docs_index_v1"
    alias.build("docs_index_v3", ["a"], {"recall_at_5": 1.0})
    assert alias.swap("docs_index_v3", {"recall_at_5": 1.0})
    assert alias.rollback()
    assert alias.current == "docs_index_v1"


def test_routing_and_explicit_fallback():
    assert route_extraction()["model"] == "lora-extractor"
    assert route_generation("What is the status update?")["model"] == "claude-haiku-4-5"
    frontier = route_generation("The customer wants a refund.")
    assert frontier["model"] == "claude-sonnet-5"
    failed = fallback_after_error(frontier)
    assert failed["fallback"] is True
    assert failed["model"] == "claude-haiku-4-5"


def test_semantic_cache_reuses_a_near_duplicate():
    cache = SemanticCache()
    cache.put("48 hour customs hold in Germany", {"draft": "status update"})
    assert cache.get("Germany customs hold 48 hour")["draft"] == "status update"
    assert cache.get("Brazil electronics refund") is None


def test_customer_pipeline_drops_the_poisoned_case_note_and_can_promise_the_clause():
    result = handle(
        "A DHL cold-chain shipment to Germany has been in an HE customs hold for 52 hours. What can we promise?",
        "held — missing commercial invoice, broker says thurs, temp reading nominal",
        "customer",
    )
    assert result["blocked_ids"] == []
    assert "rh-014" not in result["used_ids"]
    assert result["draft"]["requires_human_review"] is False
    assert "48 hours" in result["draft"]["customer_facing_draft"]


def test_a_guardrail_error_fails_closed_and_hides_the_library(monkeypatch):
    def explode(text: str):
        raise RuntimeError("scanner down")

    monkeypatch.setattr("app.pipeline.scan_and_redact", explode)
    result = handle("What can we promise after 52 hours?", "held", "customer")
    assert result["used_ids"] == []
    assert result["draft"]["requires_human_review"] is True
    assert "48 hours" not in result["draft"]["customer_facing_draft"]
    assert "jane.keller" not in result["draft"]["customer_facing_draft"]


def test_refund_request_stays_with_a_human():
    result = handle("The customer wants a refund.", "held", "customer")
    assert result["draft"]["requires_human_review"] is True
