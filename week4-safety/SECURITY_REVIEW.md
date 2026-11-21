# Security review

The customer email is assembled only after three checks. A chunk that fails a check is left out of the prompt and listed on the trace.

| Item | Severity | What it tries | Status |
|---|---|---|---|
| poison-note-01 through poison-note-10 | High | A freeform note tells the model to ignore instructions, waive the contract, or confirm a refund | Mitigated. `retrieval_guard.py` flags every one. The safety test requires all 10 to be excluded. |
| poison-chunk-01 through poison-chunk-05 | High | The same kind of instruction is hidden in a retrieved clause or case note | Mitigated. The same guard excludes all 5 before generation. |
| Case-note PII | High | A resolution note carries another customer's name, street, phone, and email | Mitigated for phone, email, and street by `pii_scan.py` before prompt assembly. A customer-facing call also never receives `resolution_history`. |
| Uncited promise | High | The draft says "we will refund" when the retrieved text does not | Mitigated. `output_moderation.py` blocks the send and the pipeline turns the draft into human review. |
| Internal notes on a customer call | High | Another customer's case history is retrieved for a customer email | Mitigated. `access_scope.py` allows a customer call to read contracts and policy only. |

Llama Guard 3 and a hosted NeMo deployment are the extra classifier layer when this moves off one machine. The pattern rail above is the check that already runs in tests, so a missing classifier does not let a poisoned chunk through.
