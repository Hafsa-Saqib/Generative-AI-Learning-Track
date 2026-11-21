import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError

from pipeline.example_selector import WordCountEmbedder, select_examples
from pipeline.generate import event_query, load_json, safe_event, submit_prompt
from pipeline.schema import ExceptionCommunication
from pipeline.semantic import find_problems


def run_event(event, chosen_snippets, chosen_history, chosen_examples) -> tuple[ExceptionCommunication, int]:
    env = Environment(loader=FileSystemLoader("templates"))
    prompt = env.get_template("communication_prompt.jinja2").render(
        current_event=safe_event(event),
        sla_snippets=chosen_snippets,
        resolution_history=chosen_history,
        few_shot_examples=chosen_examples,
    )
    allowed_ids = {item["source_id"] for item in chosen_snippets}
    attempt_prompt = prompt
    draft = None

    for attempt in range(1, 4):
        response, tool_input = submit_prompt(attempt_prompt)
        if tool_input is None:
            problems = ["The model did not submit a communication."]
        else:
            try:
                draft = ExceptionCommunication.model_validate(tool_input)
                problems = find_problems(draft, allowed_ids)
            except ValidationError as error:
                problems = [str(error)]
        if not problems:
            return draft, attempt
        attempt_prompt = (
            prompt
            + "\n\nYour previous output failed these checks:\n- "
            + "\n- ".join(problems)
            + "\nSubmit a corrected ExceptionCommunication. "
            + "Keep tool markup out of the text fields."
        )

    raw_output = "" if draft is None else draft.model_dump_json()
    fallback = ExceptionCommunication(
        exception_id=event["exception_id"],
        customer_facing_draft="The timeline is under review.",
        internal_summary=raw_output[:2000],
        grounded_commitments=[],
        citations=[],
        confidence=0.0,
        requires_human_review=True,
        review_reason="Retries exhausted. A human must read the raw model output in the internal summary.",
    )
    return fallback, 3


def main():
    events = load_json("fixtures/exception_events.json")
    snippets = load_json("fixtures/sla_snippets.json")
    history = load_json("fixtures/resolution_history.json")
    bank = load_json("fixtures/few_shot_bank.json")
    snippet_by_id = {item["source_id"]: item for item in snippets}
    history_by_id = {item["case_id"]: item for item in history}
    embedder = WordCountEmbedder([example["input"] for example in bank])

    matches = 0
    saved = []
    for event in events:
        chosen_snippets = [snippet_by_id[sid] for sid in event["snippet_ids"]]
        chosen_history = [history_by_id[cid] for cid in event["history_ids"]]
        chosen_examples = select_examples(event_query(event), bank, embedder.embed_texts, k=3)
        draft, attempts = run_event(event, chosen_snippets, chosen_history, chosen_examples)
        ok = draft.requires_human_review == event["expects_human_review"]
        matches += int(ok)
        status = "ok" if ok else "MISS"
        print(
            event["exception_id"],
            "expected review",
            event["expects_human_review"],
            "got",
            draft.requires_human_review,
            status,
            "attempts",
            attempts,
        )
        saved.append(draft.model_dump())

    print("---")
    print(matches, "of", len(events), "matched")
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/claude_results.json").write_text(
        json.dumps(saved, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()