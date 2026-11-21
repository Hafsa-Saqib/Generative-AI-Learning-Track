from pathlib import Path

from corpus_data import GOLD_DOCS, distractor_docs, expanded_contract, render_markdown

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


def write_new_files() -> list[Path]:
    written = []
    for doc in GOLD_DOCS + distractor_docs():
        path = CORPUS / doc["folder"] / doc["name"]
        if path.exists():
            continue
        path.write_text(render_markdown(doc), encoding="utf-8")
        written.append(path)
    expanded = CORPUS / "sla_contracts" / "dhl_germany_cold_chain_expanded.md"
    if not expanded.exists():
        expanded.write_text(expanded_contract(), encoding="utf-8")
        written.append(expanded)
    return written


if __name__ == "__main__":
    paths = write_new_files()
    print(f"wrote {len(paths)} new files")
    for folder in ("sla_contracts", "policy_docs", "resolution_history"):
        count = len(list((CORPUS / folder).glob("*.md")))
        print(f"{folder}: {count}")
