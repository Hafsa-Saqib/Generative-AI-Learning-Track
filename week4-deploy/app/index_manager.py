"""Content-hash skip, a new index version, an eval gate, and an alias swap."""

import hashlib


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def chunks_to_embed(previous: dict[str, str], documents: list[dict]) -> list[str]:
    changed = []
    for document in documents:
        digest = content_hash(document["text"])
        if previous.get(document["id"]) != digest:
            changed.append(document["id"])
    return changed


class IndexAlias:
    def __init__(self):
        self.versions = {}
        self.current = None
        self.previous = None
        self.log = []

    def build(self, name: str, document_ids: list[str], scores: dict) -> None:
        self.versions[name] = {"document_ids": document_ids, "scores": scores}
        self.log.append({"step": "build", "name": name, "documents": document_ids, "scores": scores})

    def swap(self, name: str, current_scores: dict) -> bool:
        candidate = self.versions[name]["scores"]
        for key, current in current_scores.items():
            if candidate.get(key, 0) < current:
                self.log.append({"step": "reject", "name": name, "key": key})
                return False
        self.previous = self.current
        self.current = name
        self.log.append({"step": "swap", "current": name, "previous": self.previous})
        return True

    def rollback(self) -> bool:
        if self.previous is None and self.current is None:
            return False
        self.current, self.previous = self.previous, self.current
        self.log.append({"step": "rollback", "current": self.current})
        return self.current is not None
