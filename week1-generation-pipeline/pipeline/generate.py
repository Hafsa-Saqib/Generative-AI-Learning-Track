import json
from pathlib import Path

from anthropic import Anthropic
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError

from pipeline.budgeter import allocate_budget, count_tokens
from pipeline.example_selector import WordCountEmbedder, select_examples
from pipeline.recover import human_review_fallback
from pipeline.schema import ExceptionCommunication
from pipeline.semantic import find_problems

MODEL = "claude-sonnet-5"
EVENT_ID = "exc_014"
TOTAL_BUDGET = 8000


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def event_query(event: dict) -> str:
    return (
        f"{event['status_code']} | {event['carrier']} | {event['country']} | "
        f"{event['commodity']} | {event['hours_in_status']} hours | {event['freeform_note']}"
    )


def safe_event(event: dict) -> dict:
    return {
        "exception_id": event["exception_id"],
        "status_code": event["status_code"],
        "carrier": event["carrier"],
        "country": event["country"],
        "commodity": event["commodity"],
        "hours_in_status": event["hours_in_status"],
        "freeform_note": event["freeform_note"],
    }


def anthropic_tool_schema() -> dict:
    schema = ExceptionCommunication.model_json_schema()

    def lock_objects(node: dict) -> None:
        for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            node.pop(keyword, None)
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}))
        for value in node.values():
            if isinstance(value, dict):
                lock_objects(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        lock_objects(item)

    lock_objects(schema)
    return schema


def submit_prompt(prompt: str):
    response = Anthropic().messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {
                "name": "submit_communication",
                "description": "Submit the finished exception communication.",
                "input_schema": anthropic_tool_schema(),
                "strict": True,
            }
        ],
        tool_choice={"type": "tool", "name": "submit_communication"},
    )
    tool_input = None
    for block in response.content:
        if block.type == "tool_use":
            tool_input = block.input
            break
    return response, tool_input


def main():
    events = load_json("fixtures/exception_events.json")
    snippets = load_json("fixtures/sla_snippets.json")
    history = load_json("fixtures/resolution_history.json")
    bank = load_json("fixtures/few_shot_bank.json")

    event = next(item for item in events if item["exception_id"] == EVENT_ID)
    snippet_by_id = {item["source_id"]: item for item in snippets}
    history_by_id = {item["case_id"]: item for item in history}

    embedder = WordCountEmbedder([example["input"] for example in bank])
    chosen = select_examples(event_query(event), bank, embedder.embed_texts, k=3)
    chosen_snippets = [snippet_by_id[sid] for sid in event["snippet_ids"]]
    chosen_history = [history_by_id[cid] for cid in event["history_ids"]]

    env = Environment(loader=FileSystemLoader("templates"))
    prompt = env.get_template("communication_prompt.jinja2").render(
        current_event=safe_event(event),
        sla_snippets=chosen_snippets,
        resolution_history=chosen_history,
        few_shot_examples=chosen,
    )

    budget = allocate_budget(
        total_budget=TOTAL_BUDGET,
        system_prompt="Ground every customer commitment in an SLA snippet.",
        current_event=event_query(event),
        sla_snippets=[item["text"] for item in chosen_snippets],
        resolution_history=[item["summary"] for item in chosen_history],
        few_shot_examples=[example["input"] for example in chosen],
    )

    allowed_ids = {item["source_id"] for item in chosen_snippets}
    attempt_prompt = prompt
    draft = None
    response = None
    accepted = False

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
            accepted = True
            print("accepted on attempt", attempt)
            break
        print("attempt", attempt, "problems:")
        for problem in problems:
            print("-", problem)
        attempt_prompt = (
            prompt
            + "\n\nYour previous output failed these checks:\n- "
            + "\n- ".join(problems)
            + "\nSubmit a corrected ExceptionCommunication. "
            + "Keep tool markup out of the text fields, and list every promise in grounded_commitments."
        )

    if not accepted:
        raw_output = "" if draft is None else draft.model_dump_json()
        draft = human_review_fallback(event["exception_id"], raw_output)
        print("fell back to human review")

    print(draft.model_dump_json(indent=2))
    print("---")
    print("local prompt tokens:", count_tokens(prompt))
    print("provider input tokens:", response.usage.input_tokens)
    print("provider output tokens:", response.usage.output_tokens)
    print("budget spent tokens:", budget["spent_tokens"])
    print("dropped examples:", len(budget["dropped_examples"]))


if __name__ == "__main__":
    main()