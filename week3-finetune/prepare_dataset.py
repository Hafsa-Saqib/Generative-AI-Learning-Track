"""Build the carrier-note dataset from the labeling rubric.

The test split is written once. A later run leaves it unchanged.
"""

import json
import random
from pathlib import Path

from schema import label_signals

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TEST_PATH = DATA / "labeled_test.jsonl"

CARRIERS = ("DHL", "FedEx", "UPS")
COMMODITIES = ("cold_chain", "electronics", "dry_goods", "pharma")

# Each spec is one situation. The renderer varies carrier, commodity, and wording.
SPECS = [
    {"subtype": "customs_hold", "reason": "customs_documentation", "date": "thu", "noun": "commercial invoice"},
    {"subtype": "customs_hold", "reason": "customs_documentation", "date": "thu", "date_conflict": True, "noun": "health certificate"},
    {"subtype": "customs_hold", "reason": "customs_inspection", "date": None, "noun": "customs exam"},
    {"subtype": "weather_delay", "reason": "weather", "date": "fri", "noun": "storm"},
    {"subtype": "weather_delay", "reason": "weather", "date": "fri", "date_hedge": True, "noun": "wind"},
    {"subtype": "delivery_exception", "reason": "receiver_closed", "date": "tue", "noun": "receiver closed"},
    {"subtype": "pickup_missed", "reason": "shipper_dock_closed", "date": "wed", "noun": "dock closed"},
    {"subtype": "address_exception", "reason": "address_incomplete", "date": None, "noun": "missing street number"},
    {"subtype": "damage", "reason": "damage", "date": "wed", "noun": "leaking carton"},
    {"subtype": "damage", "reason": "damage", "date": "fri", "noun": "crushed carton"},
    {"subtype": "breakdown", "reason": "mechanical_breakdown", "date": None, "noun": "truck breakdown"},
    {"subtype": "dangerous_goods_check", "reason": "dangerous_goods", "date": None, "noun": "lithium battery check"},
    {"subtype": "seal_issue", "reason": "seal_broken", "date": None, "noun": "broken seal"},
    {"subtype": "customs_hold", "reason": "customs_documentation", "date": None, "contradiction": True, "noun": "invoice"},
    {"subtype": "unknown", "reason": None, "date": None, "temp_problem": True, "noun": "temp high"},
    {"subtype": "unknown", "reason": None, "date": None, "no_exception": True, "noun": "departed"},
    {"subtype": "unknown", "reason": None, "date": None, "noun": "held"},
    {"subtype": "unknown", "reason": None, "date": None, "noun": "weekend clock"},
]


def _voice(carrier: str, spec: dict, commodity: str, index: int) -> str:
    noun = spec["noun"]
    day = spec.get("date")
    style = index % 3
    cold_bit = ""
    if commodity in {"cold_chain", "pharma"}:
        cold_bit = " Temp reading nominal." if not spec.get("temp_problem") else " Temp reading high, product warm."
    elif spec.get("temp_problem"):
        cold_bit = " Temp reading high."
    if spec.get("no_exception"):
        lines = [
            f"{carrier}: departed on schedule, no exception.",
            f"{carrier} dispatch — freight moved, nothing to report.",
            f"{carrier} note: all clear at the dock.",
        ]
        return lines[style]
    if noun == "held":
        return f"{carrier}: held."
    if noun == "weekend clock":
        return f"{carrier}: arrived Saturday. Does the weekend desk start the clock?"
    if spec.get("contradiction"):
        return f"{carrier}: broker says released, but the {noun} is still missing.{cold_bit}"
    if spec.get("date_conflict"):
        return f"{carrier}: held, missing {noun}. Broker says Thursday or Friday.{cold_bit}"
    if spec.get("date_hedge"):
        return f"{carrier}: {noun} hold. Maybe Friday, dispatcher is not sure.{cold_bit}"
    day_bit = ""
    if day == "thu":
        day_bit = " Broker says thurs."
    elif day == "fri" and noun in {"storm", "wind"}:
        day_bit = " Next departure Friday."
    elif day == "tue":
        day_bit = " Next attempt Tuesday."
    elif day == "wed" and noun == "dock closed":
        day_bit = " Next pickup Wednesday."
    elif day == "wed":
        day_bit = " Damage review in 2 business days."
    elif day == "fri" and noun == "crushed carton":
        day_bit = " Inspection in 3 business days."
    openings = [
        f"{carrier}: held — {noun}.{day_bit}{cold_bit}",
        f"{carrier} dispatcher note: {noun}.{day_bit}{cold_bit}",
        f"{carrier} driver: {noun}, under time pressure.{day_bit}{cold_bit}",
    ]
    return " ".join(openings[style].split())


def build_rows() -> list[dict]:
    rows = []
    number = 1
    cities = ("Hamburg", "Sao Paulo", "Lyon", "Rotterdam")
    for spec in SPECS:
        for carrier in CARRIERS:
            for commodity in COMMODITIES:
                for city in cities[:2]:
                    signals = {**spec, "commodity": commodity}
                    note = _voice(carrier, spec, commodity, number) + f" Lane {city}."
                    rows.append({
                        "id": f"n{number:03d}",
                        "carrier": carrier,
                        "commodity": commodity,
                        "note": note,
                        "label": label_signals(signals),
                    })
                    number += 1
    return rows


def _dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rows = build_rows()
    _dump(DATA / "raw_notes.jsonl", [{"id": row["id"], "carrier": row["carrier"], "note": row["note"]} for row in rows])
    order = rows[:]
    random.Random(3).shuffle(order)
    labeled = order[:300]
    splits = {
        "labeled_train.jsonl": order[:200],
        "labeled_val.jsonl": order[200:250],
    }
    for name, group in splits.items():
        _dump(DATA / name, group)
    fresh_test = order[250:300]
    if TEST_PATH.exists():
        current = [json.loads(line) for line in TEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        current_ids = {row["id"] for row in current}
        leaked = current_ids & {row["id"] for row in order[:250]}
        has_null = any(row["label"]["blocking_reason"] is None for row in current)
        if leaked or not has_null:
            _dump(TEST_PATH, fresh_test)
    else:
        _dump(TEST_PATH, fresh_test)
    test_ids = {json.loads(line)["id"] for line in TEST_PATH.read_text(encoding="utf-8").splitlines() if line}
    train_ids = {row["id"] for row in splits["labeled_train.jsonl"]}
    val_ids = {row["id"] for row in splits["labeled_val.jsonl"]}
    overlap = test_ids & (train_ids | val_ids)
    if overlap:
        raise SystemExit(f"Test ids leaked into train or val: {sorted(overlap)[:5]}")
    nulls = sum(row["label"]["blocking_reason"] is None for row in order[:300])
    print(json.dumps({
        "raw": len(rows),
        "train": 200,
        "val": 50,
        "test": len(test_ids),
        "null_labels": nulls,
        "null_fraction": round(nulls / len(labeled), 3),
    }))


if __name__ == "__main__":
    main()
