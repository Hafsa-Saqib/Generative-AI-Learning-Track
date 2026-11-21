"""Fields extracted from a freeform carrier note.

The rubric in labeling_rubric.md is the only definition of a correct label.
"""

from pydantic import BaseModel, field_validator

SUBTYPES = (
    "customs_hold",
    "weather_delay",
    "delivery_exception",
    "pickup_missed",
    "damage",
    "address_exception",
    "breakdown",
    "dangerous_goods_check",
    "seal_issue",
    "unknown",
)

BLOCKING_REASONS = (
    "customs_documentation",
    "customs_inspection",
    "weather",
    "receiver_closed",
    "shipper_dock_closed",
    "address_incomplete",
    "damage",
    "mechanical_breakdown",
    "dangerous_goods",
    "seal_broken",
)

WEEKDAYS = {
    "tue": "2026-11-03",
    "wed": "2026-11-04",
    "thu": "2026-11-05",
    "fri": "2026-11-06",
}


class ExtractionResult(BaseModel):
    exception_subtype: str
    blocking_reason: str | None = None
    estimated_resolution_date: str | None = None
    requires_cold_chain_monitoring: bool

    @field_validator("exception_subtype")
    @classmethod
    def known_subtype(cls, value: str) -> str:
        if value not in SUBTYPES:
            raise ValueError(f"Unknown subtype: {value}")
        return value

    @field_validator("blocking_reason")
    @classmethod
    def known_reason(cls, value: str | None) -> str | None:
        if value is not None and value not in BLOCKING_REASONS:
            raise ValueError(f"Unknown blocking reason: {value}")
        return value

    @field_validator("estimated_resolution_date")
    @classmethod
    def known_date(cls, value: str | None) -> str | None:
        allowed = set(WEEKDAYS.values())
        if value is not None and value not in allowed:
            raise ValueError(f"Date is outside the rubric week: {value}")
        return value


def label_signals(signals: dict) -> dict:
    """Apply the written rubric. Every labeled row is produced through this function."""
    if signals.get("no_exception"):
        return {
            "exception_subtype": "unknown",
            "blocking_reason": None,
            "estimated_resolution_date": None,
            "requires_cold_chain_monitoring": False,
        }
    reason = signals.get("reason")
    if reason is not None and reason not in BLOCKING_REASONS:
        raise ValueError(f"Unknown blocking reason: {reason}")
    subtype = signals["subtype"]
    if subtype not in SUBTYPES:
        raise ValueError(f"Unknown subtype: {subtype}")
    if signals.get("date_conflict") or signals.get("date_hedge"):
        resolution = None
    else:
        token = signals.get("date")
        resolution = None if token is None else WEEKDAYS[token]
    commodity = signals.get("commodity", "dry_goods")
    cold = commodity in {"cold_chain", "pharma"} or bool(signals.get("temp_problem"))
    return {
        "exception_subtype": subtype,
        "blocking_reason": reason,
        "estimated_resolution_date": resolution,
        "requires_cold_chain_monitoring": cold,
    }
