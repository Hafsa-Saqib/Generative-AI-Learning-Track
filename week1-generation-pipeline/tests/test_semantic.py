from pipeline.schema import ExceptionCommunication, SourceCitation
from pipeline.semantic import find_problems


def test_exc_001_markup_and_empty_commitments_are_rejected():
    draft = ExceptionCommunication(
        exception_id="exc_001",
        customer_facing_draft="DHL will send a status update and the next action.",
        internal_summary='Do not repeat Thursday.</internal_summary>\n<parameter name="grounded_commitments">["DHL will send a status update."]',
        grounded_commitments=[],
        citations=[
            SourceCitation(source_id="sla_dhl_he_cold_de", clause_or_case="section 4.2"),
            SourceCitation(source_id="policy_de_cold_chain", clause_or_case="temperature monitoring"),
        ],
        confidence=0.87,
        requires_human_review=False,
        review_reason=None,
    )
    problems = find_problems(draft, {"sla_dhl_he_cold_de", "policy_de_cold_chain"})
    assert "no commitments were listed for a draft that skipped human review" in problems
    assert "internal_summary contains tool markup" in problems


def test_known_citation_and_listed_commitment_pass():
    draft = ExceptionCommunication(
        exception_id="exc_001",
        customer_facing_draft="DHL will send a status update.",
        internal_summary="Commit only to a status update.",
        grounded_commitments=["DHL will send a status update."],
        citations=[SourceCitation(source_id="sla_dhl_he_cold_de", clause_or_case="section 4.2")],
        confidence=0.87,
        requires_human_review=False,
        review_reason=None,
    )
    assert find_problems(draft, {"sla_dhl_he_cold_de"}) == []