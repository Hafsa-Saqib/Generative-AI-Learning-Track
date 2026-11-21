import json
import math
from pathlib import Path


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_size = math.sqrt(sum(a * a for a in left))
    right_size = math.sqrt(sum(b * b for b in right))
    if left_size == 0 or right_size == 0:
        return 0.0
    return dot / (left_size * right_size)


def select_examples(
    query_text: str,
    bank: list[dict],
    embed_texts,
    k: int = 3,
    duplicate_limit: float = 0.97,
) -> list[dict]:
    if not bank:
        return []

    query_vector = embed_texts([query_text])[0]
    example_vectors = embed_texts([example["input"] for example in bank])
    ranked = sorted(
        range(len(bank)),
        key=lambda index: cosine_similarity(query_vector, example_vectors[index]),
        reverse=True,
    )

    chosen = []
    skipped = []
    for index in ranked:
        if len(chosen) == k:
            break
        too_similar = any(
            cosine_similarity(example_vectors[index], example_vectors[kept]) > duplicate_limit
            for kept in chosen
        )
        if too_similar:
            skipped.append(index)
            continue
        chosen.append(index)

    for index in skipped:
        if len(chosen) == k:
            break
        chosen.append(index)

    return [bank[index] for index in chosen]


def cached_embed(texts: list[str], embed_texts, cache_path: str, model_name: str) -> list[list[float]]:
    path = Path(cache_path)
    payload = {"model": model_name, "vectors": {}}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if loaded.get("model") == model_name:
            payload = loaded

    vectors = payload["vectors"]
    missing = [text for text in texts if text not in vectors]
    if missing:
        fresh = embed_texts(missing)
        for text, vector in zip(missing, fresh):
            vectors[text] = vector
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    return [vectors[text] for text in texts]


class WordCountEmbedder:
    def __init__(self, known_texts: list[str]):
        words = set()
        for text in known_texts:
            words.update(text.lower().split())
        self.vocabulary = sorted(words)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            counts = dict.fromkeys(self.vocabulary, 0)
            for word in text.lower().split():
                if word in counts:
                    counts[word] += 1
            vectors.append([float(counts[word]) for word in self.vocabulary])
        return vectors