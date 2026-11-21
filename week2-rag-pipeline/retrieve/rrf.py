def reciprocal_rank_fusion(rankings: list[list], k: int = 60) -> list:
    """Merge ranked lists by rank. The score is the sum of 1 / (k + rank)."""
    scores: dict = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
