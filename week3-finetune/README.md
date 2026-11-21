# Week 3 — when to fine-tune

Fine-tuning is for a narrow behavior that has to run on every carrier note. It does not replace the Week 2 library. The note is turned into four fields. The contract clauses still come from search.

## Dataset

`prepare_dataset.py` writes 432 raw notes and a 200 / 50 / 50 labeled split. The rubric is `labeling_rubric.md`. At least 10% of labels have no blocking reason. The test file is written once and is not rewritten.

Carriers are DHL, FedEx, and UPS. Notes include missing documents, exams, weather, closed receivers, closed docks, damage, breakdowns, dangerous goods, broken seals, contradictory broker comments, and notes with no exception.

## Local LoRA, at no host cost

`finetune/submit_job.py` trains a LoRA adapter on this PC with Hugging Face PEFT. The base model is SmolLM2 135M. Its weights stay frozen. Rank 8 matrices are the only trained weights, and the training cost is $0.

```powershell
..\week1-generation-pipeline\.venv\Scripts\python.exe finetune\submit_job.py
```

The adapter is written to `outputs\lora-adapter`. `finetune\poll_status.py` prints `COMPLETED` once that folder exists. A paid host is not required.

`eval/score_local.py` scores the same 50 held-out notes on the untuned model and on the adapter. The measured result is 4.5% untuned and 5.5% with the adapter. The only field either model matched was cold-chain monitoring (18% untuned, 22% with the adapter). Subtype, blocking reason, and date were 0%. Zero replies were a complete rubric JSON, so a person still reads the note. Inference cost is $0.

## Trace

`observability/corrupt_and_trace.py` puts a wrong "4 hours" clause into a retrieved chunk in memory. The finder reads only the trace and names `retrieval.dense`. Postgres is not opened and is not changed.

```powershell
..\week1-generation-pipeline\.venv\Scripts\python.exe observability\corrupt_and_trace.py
```

The command prints the stage it found and writes `outputs/bad_retrieval_trace.json`. The same check is `tests/test_eval_and_trace.py::test_trace_alone_finds_the_bad_sla_number`.
