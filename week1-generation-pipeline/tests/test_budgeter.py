import tiktoken
from pipeline.budgeter import allocate_budget

ENC = tiktoken.get_encoding("cl100k_base")


def words(n: int) -> str:
    text = "token " * (n * 2)
    return ENC.decode(ENC.encode(text)[:n])


def test_large_budget_keeps_everything():
    examples = ["best", "middle", "weak"]
    history = ["old case", "new case"]
    snippets = ["sla one", "sla two"]
    result = allocate_budget(10000, "rules", "event", snippets, history, examples)
    assert result["few_shot_examples"] == examples
    assert result["resolution_history"] == history
    assert result["sla_snippets"] == snippets
    assert result["dropped_examples"] == []
    assert result["dropped_history"] == []
    assert result["over_budget"] is False


def test_tight_budget_drops_extra_examples_before_sla():
    snippets = [words(10), words(10)]
    history = [words(10)]
    examples = [words(10), words(50), words(50)]
    result = allocate_budget(70, words(10), words(10), snippets, history, examples)
    assert result["few_shot_examples"] == [examples[0]]
    assert result["sla_snippets"] == snippets
    assert result["resolution_history"] == history
    assert examples[1] in result["dropped_examples"]
    assert examples[2] in result["dropped_examples"]


def test_never_drops_the_last_example():
    examples = [words(40), words(40), words(40)]
    result = allocate_budget(10, words(20), words(20), [words(20)], [], examples)
    assert result["few_shot_examples"] == [examples[0]]
    assert result["system_prompt"] == words(20)
    assert result["current_event"] == words(20)


def test_history_is_removed_before_sla_snippets():
    history = [words(40), words(5)]
    snippets = [words(5), words(5)]
    result = allocate_budget(38, words(5), words(5), snippets, history, [words(5)])
    assert result["resolution_history"] == [history[1]]
    assert result["dropped_history"] == [history[0]]
    assert result["sla_snippets"] == snippets
    assert result["dropped_snippets"] == []