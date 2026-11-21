# Week 4 — safety and deployment

Week 4 wraps the Week 1 email and the Week 2 library.

1. A customer call can retrieve contracts and policy. It cannot retrieve resolution history.
2. Every remaining chunk is scanned for an embedded instruction and for a phone number, email, or street address.
3. The draft is the Week 1 shape. A promise is copied only from retrieved text. An uncited commitment is not sent.
4. Extraction is routed to the small LoRA adapter. A routine status question is routed to Haiku. A refund, same-day, or release question is routed to Sonnet. A provider error is recorded as an explicit fallback.
5. A new index is built beside the current one. It goes live only when recall holds. The previous name stays available for rollback. Unchanged documents are not embedded again.

The container file is `Dockerfile`. Compose starts Postgres and the app with a read-only root filesystem. The CI workflow runs the Week 1, 2, 3, and 4 tests and the offline score gate.

Findings and remediation are in `week4-safety/SECURITY_REVIEW.md`.
