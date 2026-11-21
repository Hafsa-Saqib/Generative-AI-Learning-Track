# Shipment emails that stay inside the contract

A stuck shipment comes in. The system writes the customer email only from a contract sentence it found. If it cannot find one, a person checks the email.

Open `interface/index.html` in a browser. The four buttons are the four weeks.

The vector database is PostgreSQL with pgvector. Each clause is one row: the words and the meaning numbers are stored together.

```
Exception input
    |
    v
PII redaction and injection guard          week4-safety
    |
    v
Hybrid retrieval                           week2-rag-pipeline
    pgvector dense + sparse + rerank
    |
    v
Token-budgeted context assembly            week1 budgeter
    |
    v
Multi-provider generation                  Claude and Groq
    |
    v
Schema check and policy grounding gate
    |
    +-- clause supports the promise --> customer draft
    |
    +-- no supporting clause, bad JSON,
        injection, or guardrail error --> person checks the email
```

Fine-tuning does not sit on this path. It was tried only for reading a carrier note into four fields. Contract facts stay in retrieval.

## Repository map

| Path | What it holds |
|---|---|
| `week1-generation-pipeline` | Prompt template, few-shot picker, token budget, JSON recovery, Claude and Groq drafts |
| `week2-rag-pipeline` | 60 documents, 401 chunks, pgvector hybrid search, 30 held-out questions |
| `week3-finetune` | Note rubric, local LoRA comparison, in-memory trace of a planted bad clause |
| `week4-safety` | Injection filter, PII masking, customer/internal access split |
| `week4-deploy` | Request path, routing, cache, index alias swap, fail-closed guard |
| `interface` | The page used in the presentation. `build_view.py` writes `index.html` |
| `.github/workflows` | `eval-gate.yml` on a pull request, `index-rebuild.yml` when started by hand |

## Measured choices

The customer email stays prompt-only, with the contract sentence coming from retrieval. Four options were weighed:

| Option | What was measured |
|---|---|
| Prompt only, no retrieved clause | The draft says the timeline is under review and a person checks it. Confidence on that path is 0.34. |
| Prompt plus hybrid retrieval | Recall@5 is 30/30 (100%). The earlier dense-only LlamaIndex run on the same chunks was 28/30 (93%). Context precision at 5 is 0.24: the answering sentence is usually one of the five chunks. |
| Long context, whole library in the prompt | Not run. The held-out search already returns the one clause the email is allowed to quote. |
| Fine-tune the note reader | Local SmolLM2 LoRA, $0. Untuned accuracy 4.5% and median 18.6s. Adapter accuracy 5.5% and median 9.4s. Neither reply is a complete rubric JSON, so a person still reads the note. |

Week 4 blocked 15 of 15 planted prompt injections. A customer call cannot read resolution history. Phone, email, and street are masked before a draft is written. If a guardrail raises an error, the library is hidden and the email stays with a person.

## Trace: the planted "4 hours" clause

The live library says 48 hours. The trace demo puts a wrong "4 hours" sentence in memory and asks the trace, not Postgres, where it first appears.

```powershell
Set-Location .\week3-finetune
..\week1-generation-pipeline\.venv\Scripts\python.exe observability\corrupt_and_trace.py
```

The command prints `{"stage": "retrieval.dense", "trace_id": "..."}` and writes `week3-finetune/outputs/bad_retrieval_trace.json`. The same assertion is `tests/test_eval_and_trace.py::test_trace_alone_finds_the_bad_sla_number`.

## Check

Copy `.env.example` to `.env` only when you call a paid API. The commands below do not read those keys. Use the Week 1 virtualenv.

```powershell
$py = ".\week1-generation-pipeline\.venv\Scripts\python.exe"
Set-Location .\week1-generation-pipeline; & $py -m pytest -q
Set-Location ..\week2-rag-pipeline; & $py -m pytest -q
Set-Location ..\week3-finetune; & $py -m pytest -q
& $py eval\run_evals.py
Set-Location ..\week4-safety; & $py -m pytest -q
Set-Location ..\week4-deploy; & $py -m pytest -q
```

`.github/workflows/eval-gate.yml` runs that list on a pull request. `.github/workflows/index-rebuild.yml` runs the same list when someone starts it by hand. Both refuse to pass when Recall@5, faithfulness, or the human-review decision on the saved Week 2 file drops below 1.0, or when the Week 3 split is not 200 / 50 / 50.

The Week 2 container is `week2-rag-pipeline-postgres-1`. Starting that same container again, without creating a new volume, brought back 401 chunks. All 401 are embedded with `voyage-3`.
