import math
from pipeline.example_selector import cached_embed, cosine_similarity, select_examples


def test_cosine_of_matching_and_unrelated_vectors():
    assert math.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0)


def test_selector_skips_a_near_duplicate():
    bank = [
        {"example_id": "A", "input": "best"},
        {"example_id": "B", "input": "copy"},
        {"example_id": "C", "input": "different"},
        {"example_id": "D", "input": "also different"},
    ]
    vectors = {
        "query": [1.0, 0.0],
        "best": [1.0, 0.0],
        "copy": [0.995, 0.1],
        "different": [0.6, 0.8],
        "also different": [0.2, 0.98],
    }

    def embed_texts(texts):
        return [vectors[text] for text in texts]

    chosen = select_examples("query", bank, embed_texts, k=3)
    assert [item["example_id"] for item in chosen] == ["A", "C", "D"]


def test_cache_embeds_each_text_once(tmp_path):
    calls = []

    def embed_texts(texts):
        calls.append(list(texts))
        return [[float(len(text)), 0.0] for text in texts]

    cache_path = tmp_path / "embeddings.json"
    first = cached_embed(["alpha", "beta"], embed_texts, cache_path, "word-count")
    second = cached_embed(["alpha", "beta"], embed_texts, cache_path, "word-count")
    assert first == second
    assert calls == [["alpha", "beta"]]