from pipeline.schema import ExceptionCommunication


def find_problems(draft: ExceptionCommunication, allowed_source_ids: set[str]) -> list[str]:
    problems = []

    for citation in draft.citations:
        if citation.source_id not in allowed_source_ids:
            problems.append(f"unknown source_id {citation.source_id}")

    if draft.requires_human_review and not draft.review_reason:
        problems.append("human review is missing a reason")

    if not draft.requires_human_review and not draft.grounded_commitments:
        problems.append("no commitments were listed for a draft that skipped human review")

    for field_name in ("customer_facing_draft", "internal_summary"):
        text = getattr(draft, field_name)
        if "<parameter" in text or "</" in text:
            problems.append(f"{field_name} contains tool markup")

    return problems