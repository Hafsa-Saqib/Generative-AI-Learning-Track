"""Reuse a recent answer when the new question uses almost the same words."""


def _tokens(text: str) -> set[str]:
    return {token.strip(".,?").lower() for token in text.split() if token.strip(".,?")}


def similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SemanticCache:
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.rows = []
        self.prompt_prefixes = {}

    def remember_prompt(self, name: str, text: str) -> None:
        self.prompt_prefixes[name] = text

    def get(self, query: str):
        for row in self.rows:
            if similarity(query, row["query"]) >= self.threshold:
                return row["result"]
        return None

    def put(self, query: str, result: dict) -> None:
        self.rows.append({"query": query, "result": result})
