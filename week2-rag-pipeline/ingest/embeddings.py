"""1024-dimension embeddings.

Voyage is used when VOYAGE_API_KEY is set, with input_type document for stored
chunks and input_type query for the question. Otherwise a hashed TF-IDF vector
of the same width fills the pgvector column.
"""

import hashlib
import math
import os
import re
import time

from chunkers import count_tokens

DIMS = 1024
LOCAL_MODEL = "hash-tfidf-1024-v1"
VOYAGE_MODEL = "voyage-3"
TOKEN = re.compile(r"[a-z0-9]+")
# Free Voyage accounts allow 3 requests and 10,000 tokens per minute.
VOYAGE_BATCH_TOKENS = 2500
VOYAGE_BATCH_ITEMS = 64
VOYAGE_PAUSE_SECONDS = 21


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def fit_idf(texts: list[str]) -> dict[str, float]:
    document_count = len(texts) or 1
    frequencies: dict[str, int] = {}
    for text in texts:
        for token in set(tokens(text)):
            frequencies[token] = frequencies.get(token, 0) + 1
    return {
        token: math.log((document_count + 1) / (frequency + 1)) + 1
        for token, frequency in frequencies.items()
    }


def _hash_vector(text: str, idf: dict[str, float]) -> list[float]:
    vector = [0.0] * DIMS
    counts: dict[str, int] = {}
    for token in tokens(text):
        counts[token] = counts.get(token, 0) + 1
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode()).digest()
        bucket = int.from_bytes(digest[:4], "little") % DIMS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = (1.0 + math.log(count)) * idf.get(token, 1.0)
        vector[bucket] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _voyage_batches(texts: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        text_tokens = count_tokens(text)
        over_budget = current and (
            current_tokens + text_tokens > VOYAGE_BATCH_TOKENS or len(current) >= VOYAGE_BATCH_ITEMS
        )
        if over_budget:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += text_tokens
    if current:
        batches.append(current)
    return batches


def _embed_voyage(texts: list[str], input_type: str) -> list[list[float]]:
    import voyageai
    from voyageai.error import RateLimitError

    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    batches = _voyage_batches(texts)
    vectors: list[list[float]] = []
    for index, batch in enumerate(batches, start=1):
        if index > 1:
            time.sleep(VOYAGE_PAUSE_SECONDS)
        print(f"Voyage batch {index} of {len(batches)} ({len(batch)} chunks)", flush=True)
        for attempt in range(4):
            try:
                result = client.embed(batch, model=VOYAGE_MODEL, input_type=input_type)
                break
            except RateLimitError:
                if attempt == 3:
                    raise
                print("Voyage rate limit. Waiting 60 seconds.", flush=True)
                time.sleep(60)
        vectors.extend(result.embeddings)
    return vectors


def embed_texts(texts: list[str], input_type: str, idf: dict[str, float] | None = None) -> tuple[list[list[float]], str]:
    if input_type not in {"document", "query"}:
        raise ValueError("input_type must be document or query")
    if os.environ.get("VOYAGE_API_KEY"):
        return _embed_voyage(texts, input_type), VOYAGE_MODEL
    weights = idf or {}
    return [_hash_vector(text, weights) for text in texts], LOCAL_MODEL
