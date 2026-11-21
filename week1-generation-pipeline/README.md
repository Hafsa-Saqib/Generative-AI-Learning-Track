# Week 1 — Exception communication drafts

This pipeline turns a shipment exception plus contract snippets into a structured customer draft. A promise is allowed only when a snippet supports it. Otherwise the draft is marked for a human.

## What was built

- `pipeline/schema.py` defines the draft shape.
- `templates/communication_prompt.jinja2` fills the instructions, the shipment, the snippets, the old cases, and three examples.
- `pipeline/example_selector.py` picks those three examples by word overlap.
- `pipeline/budgeter.py` drops extra examples first and protects contract text. Its tests pass. The number it prints is lower than Claude's bill, because it counts a short stand-in rather than the full prompt. Claude's input counts on this run were about 2,200 to 2,600 tokens. The budgeter often printed about 200 to 300.
- `pipeline/generate.py` calls Claude and retries when `pipeline/semantic.py` finds a problem.
- `outputs/claude_results.json` holds all 15 drafts.

## Scoreboard

14 of 15 matched the answer key.

The three shipments with no supported promise all went to a human: exc_002 (Brazil electronics), exc_011 (refund after a breakdown), and exc_014 (same-day release after only 10 hours).

exc_008 was the miss. The shipment had only been at the gateway for 2 hours. Claude tried to promise temperature monitoring, but the reply kept putting tool markup in the summary. After three rejects, the program sent it to a human. The answer key had expected no review. The markup check caused the miss. The temperature promise was also weak: the only snippet on that shipment is the 48-hour response clause, which does not say temperature monitoring will continue.

## Checkpoint

The first exc_001 reply looked convincing and was valid JSON. The promises were trapped inside `internal_summary` as tool markup, and `grounded_commitments` was empty. Nothing called the checker, so the reply was printed. Wiring `find_problems` into the retry loop caught the same markup on later shipments.

## Industry variations

Retail replenishment uses supplier minimum-order terms instead of a carrier SLA. The draft must not state a reorder quantity the contract does not support.

Manufacturing supplier reports use inspection logs. The draft must not name a defect cause unless an inspection finding is cited.

Supply-chain sourcing uses contract clauses. A misquoted obligation is a legal risk, so the citation check matters most there.

## Claude vs Groq

The second model was Groq `openai/gpt-oss-20b`, not GPT-5.2. Both used the same 15 shipments, the same template, and the same checker.

They agreed on 14 shipments. On exc_002, exc_011, and exc_014, both refused to make a promise and set requires_human_review to true.

They differed on exc_008. Claude’s final saved draft asked for a human because three replies contained tool markup and the retries ran out. Groq returned a draft on attempt 1 with requires_human_review false. Its only commitment was conditional: a status update is owed if the customs hold later exceeds 48 hours. It cited sla_dhl_he_cold_de and did not promise a release time.