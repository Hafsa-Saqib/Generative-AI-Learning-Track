# Week 2 — retrieval

The library has 60 documents and 401 chunks in Postgres (`document_chunks`). Search is hybrid: a 1024-dimension vector over an HNSW index (`m = 16`, `ef_construction = 64`) plus an English `tsvector` with a GIN index. The two lists are merged with reciprocal rank fusion (`k = 60`), then the top 20 are cut to 5.

## Scores on 30 held-out questions

| Measure | Result |
|---|---|
| Recall at 5 | 30/30 |
| Context precision at 5 | 0.24 |
| Average precision | 0.87 |
| Faithfulness | 1.00 |
| Human-review decision | 30/30 |
| Embedding | voyage-3 |
| Reranker | cohere-rerank-v3.5 |

Context precision is 0.24. Each question returns five chunks, and the answering sentence is usually in one of them. On five questions the Claude situating note repeats that sentence, so a second or third chunk also counts. Average precision is 0.87, so the answering chunk is usually at or near the first rank.

Faithfulness means every customer promise is a sentence copied from a retrieved chunk. Questions with no supported promise go to human review and add no promise.

## What ran in this session

Stored chunks were embedded with Voyage `voyage-3` and `input_type=document`. The 30 questions were embedded in one call with `input_type=query`. Cohere `rerank-v3.5` cut the fused top 20 down to 5. The free Voyage account allows 3 calls and 10,000 tokens a minute, and the Cohere trial key allows 10 calls a minute, so the script pauses between those calls.

All 401 situating notes were written by Claude. The Germany 4.2.2 note from the first live call was kept. The chunks were embedded again with Voyage `voyage-3` after those notes were stored.

The long warehouse appendix is 1,976 tokens. Two chunks sent with the same cached prefix used 3,095 cache-write tokens on the first call and 3,095 cache-read tokens on the second. The short Germany contract remains the earlier call: 410 input tokens, cache write 0, cache read 0.

`eval/run_judges.py` asks Claude, through RAGAS and through DeepEval, whether each customer draft is supported by the retrieved clauses. On the Claude notes the means are RAGAS faithfulness 0.383 and DeepEval faithfulness 0.967. The judge model is `claude-sonnet-5`.

DeepEval accepted 29 of the 30 drafts. It scored q14 at 0. That draft says “The timeline is under review.” because the customer asked when a chilled-food customs exam ends, and no retrieved sentence supports that prediction. RAGAS is lower because it splits a draft into claims. Most review drafts score 0. A few promise drafts score part of the way: q05 at 0.50, and q08, q11, and q27 at 0.67. The local faithfulness check stays at 1.00.

`ingest/contextualize_corpus.py` asks Claude for a situating note on every chunk that does not already have one. The Germany 4.2.2 note from the first live call is kept. After that script finishes, reload and rescore with the Voyage key still set:

```powershell
..\week1-generation-pipeline\.venv\Scripts\python.exe ingest\contextualize_corpus.py
..\week1-generation-pipeline\.venv\Scripts\python.exe ingest\load_chunks.py
..\week1-generation-pipeline\.venv\Scripts\python.exe eval\run_eval.py
..\week1-generation-pipeline\.venv\Scripts\python.exe eval\run_judges.py
```

An earlier LlamaIndex rebuild used dense search only and the local embedder. It recalled 28 of 30. This Voyage plus Cohere hybrid run recalled 30 of 30.
