"""Final check before a customer email can be sent."""

PROMISE_WORDS = ("will ", "refund", "within ", "same-day", "same day")


def moderate(draft: dict, allowed_text: str) -> dict:
    email = draft.get("customer_facing_draft", "")
    promises = draft.get("grounded_commitments") or []
    unsupported = [promise for promise in promises if promise not in allowed_text]
    commitment_language = any(word in email.lower() for word in PROMISE_WORDS)
    cited = bool(draft.get("citations"))
    blocked = bool(unsupported) or (commitment_language and not cited and not draft.get("requires_human_review"))
    reason = None
    if unsupported:
        reason = "A promise is not present in the retrieved text."
    elif blocked:
        reason = "The email uses commitment language without a citation."
    return {"allowed": not blocked, "reason": reason}
