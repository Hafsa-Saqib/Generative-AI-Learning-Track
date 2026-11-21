"""Cheap model for the narrow extractor. Frontier model only for a high-stakes draft."""

HIGH_STAKES = ("refund", "same-day", "same day", "customs release")


def route_extraction() -> dict:
    return {"task": "extraction", "model": "lora-extractor", "reason": "Narrow field extraction runs on the small adapter."}


def route_generation(question: str) -> dict:
    lowered = question.lower()
    if any(phrase in lowered for phrase in HIGH_STAKES):
        return {"task": "generation", "model": "claude-sonnet-5", "reason": "The question asks for a high-stakes commitment."}
    return {"task": "generation", "model": "claude-haiku-4-5", "reason": "A routine status question stays on the cheap model."}


def fallback_after_error(route: dict) -> dict:
    return {
        **route,
        "model": "claude-haiku-4-5",
        "fallback": True,
        "reason": "The chosen provider returned an error. The fallback model is recorded, not swapped silently.",
    }
