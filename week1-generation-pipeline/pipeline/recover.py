"""Read a provider reply without calling the provider.

Claude submits a tool object. Groq submits a JSON string, sometimes inside a
fence. A reply that is not valid, or that still fails the checks after the
caller retries, becomes a human review.
"""

from pydantic import ValidationError

from pipeline.schema import ExceptionCommunication
from pipeline.semantic import find_problems


def parse_model_json(raw: str) -> ExceptionCommunication:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return ExceptionCommunication.model_validate_json(text)


def inspect_payload(payload: dict, allowed_ids: set[str]) -> tuple[ExceptionCommunication | None, list[str]]:
    try:
        draft = ExceptionCommunication.model_validate(payload)
    except ValidationError as error:
        return None, [str(error)]
    return draft, find_problems(draft, allowed_ids)


def human_review_fallback(exception_id: str, raw_output: str = "") -> ExceptionCommunication:
    return ExceptionCommunication(
        exception_id=exception_id,
        customer_facing_draft="The timeline is under review.",
        internal_summary=raw_output[:2000],
        grounded_commitments=[],
        citations=[],
        confidence=0.0,
        requires_human_review=True,
        review_reason="Retries exhausted. A human must read the raw model output in the internal summary.",
    )


def unsupported_commitments(commitments: list[str], clause_text: str) -> list[str]:
    """A commitment may be sent only when that exact sentence is in a retrieved clause."""
    return [item for item in commitments if item not in clause_text]
