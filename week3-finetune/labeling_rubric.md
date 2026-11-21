# Labeling rubric

One curator uses this page for every row. A second reading of the same note must produce the same four fields.

## exception_subtype

Use the closest operational bucket, not the customer's request.

| Subtype | Use it when the note describes |
|---|---|
| customs_hold | A customs or broker hold |
| weather_delay | Weather stopping a departure or delivery |
| delivery_exception | The receiver could not take the freight |
| pickup_missed | The collection or pickup did not happen |
| damage | Leak, crush, or refusal for condition |
| address_exception | A missing or wrong address detail |
| breakdown | A vehicle or equipment failure |
| dangerous_goods_check | A lithium or dangerous-goods hold |
| seal_issue | A broken or mismatched seal |
| unknown | No exception, or too little text to choose |

## blocking_reason

This is why the freight is stuck. It is null when the note does not name a block.

| Reason | Counts as this | Does not count as this |
|---|---|---|
| customs_documentation | Missing invoice, certificate, or broker document | A physical exam with no missing document |
| customs_inspection | Exam, scan, or inspection | A missing-paper hold |
| weather | Storm, wind, or cancelled departure for weather | A late truck with no weather |
| receiver_closed | Receiver, consignee, or building closed | The shipper's own dock |
| shipper_dock_closed | The shipper dock was closed for pickup | A closed receiver |
| address_incomplete | Missing street, number, or apartment | A bad phone number alone |
| damage | Leak, crush, torn carton, refused for condition | A temperature comment with no damage |
| mechanical_breakdown | Truck, trailer, or reefer unit failed | A missed pickup with no failure |
| dangerous_goods | Battery, DG, or dangerous-goods check | Ordinary customs |
| seal_broken | Seal broken, missing, or mismatched | An open carton with no seal mentioned |

If the note says the freight is released and also says a document is still missing, the document is still the block.

## estimated_resolution_date

The anchor week starts Monday 2026-11-02.

| Words in the note | Date |
|---|---|
| Tuesday, or 2 business days from Monday | 2026-11-03 |
| Wednesday | 2026-11-04 |
| Thursday, or "thurs" | 2026-11-05 |
| Friday, or 3 business days from Monday | 2026-11-06 |

Leave the date null when the note gives two different days, says "maybe", "or", or gives no day. A customer asking for a refund is not a date.

## requires_cold_chain_monitoring

True when the note is about cold-chain or pharma freight, or when it reports a temperature problem ("warm", "temp high"). A dry shipment that says "temp nominal" stays false. "Temp nominal" on a cold-chain shipment stays true, because the freight still needs monitoring.

## Null rows

At least 10% of labels have `blocking_reason: null`. "All clear", "departed", "held" with no cause, and a weekend-clock question are nulls. Do not guess a reason.
