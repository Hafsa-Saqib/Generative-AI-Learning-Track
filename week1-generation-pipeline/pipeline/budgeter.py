import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def trim_to_tokens(text: str, max_tokens: int) -> str:
    tokens = ENC.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return ENC.decode(tokens[:max_tokens])


def allocate_budget(
    total_budget: int,
    system_prompt: str,
    current_event: str,
    sla_snippets: list[str],
    resolution_history: list[str],
    few_shot_examples: list[str],
) -> dict:
    usable = total_budget - (total_budget // 10)
    examples = list(few_shot_examples)
    history = list(resolution_history)
    snippets = list(sla_snippets)

    dropped_examples = []
    dropped_history = []
    truncated_history = []
    dropped_snippets = []
    truncated_snippets = []

    def spent() -> int:
        chunks = [system_prompt, current_event, *examples, *history, *snippets]
        return sum(count_tokens(chunk) for chunk in chunks)

    while spent() > usable and len(examples) > 1:
        removed = examples.pop()
        dropped_examples.append(removed)
        if count_tokens(removed) == 0:
            break

    while spent() > usable and len(history) > 1:
        dropped_history.append(history.pop(0))

    if spent() > usable and history:
        original = history[-1]
        shortened = trim_to_tokens(original, 30)
        if count_tokens(shortened) < count_tokens(original):
            history[-1] = shortened
            truncated_history.append(
                {
                    "from_tokens": count_tokens(original),
                    "to_tokens": count_tokens(shortened),
                }
            )
        if spent() > usable:
            dropped_history.append(history.pop())

    while spent() > usable and len(snippets) > 1:
        removed = snippets.pop()
        dropped_snippets.append(removed)
        if count_tokens(removed) == 0:
            break

    if spent() > usable and snippets:
        original = snippets[0]
        shortened = trim_to_tokens(original, 30)
        if count_tokens(shortened) < count_tokens(original):
            snippets[0] = shortened
            truncated_snippets.append(
                {
                    "from_tokens": count_tokens(original),
                    "to_tokens": count_tokens(shortened),
                }
            )

    return {
        "usable_budget": usable,
        "spent_tokens": spent(),
        "system_prompt": system_prompt,
        "current_event": current_event,
        "few_shot_examples": examples,
        "resolution_history": history,
        "sla_snippets": snippets,
        "dropped_examples": dropped_examples,
        "dropped_history": dropped_history,
        "truncated_history": truncated_history,
        "dropped_snippets": dropped_snippets,
        "truncated_snippets": truncated_snippets,
        "over_budget": spent() > usable,
    }