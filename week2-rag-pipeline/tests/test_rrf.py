from retrieve.rrf import reciprocal_rank_fusion


def test_higher_rank_beats_a_later_list():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert fused[0] == "b"
    assert fused.index("a") < fused.index("c")


def test_empty_lists():
    assert reciprocal_rank_fusion([[], []], k=60) == []
