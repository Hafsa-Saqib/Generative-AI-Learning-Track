"""A customer-facing call may not read internal case notes."""

CUSTOMER_VISIBLE = {"sla_contract", "policy_doc"}


def visible_chunks(chunks: list[dict], caller: str) -> list[dict]:
    if caller == "internal":
        return list(chunks)
    return [chunk for chunk in chunks if chunk.get("document_type") in CUSTOMER_VISIBLE]
