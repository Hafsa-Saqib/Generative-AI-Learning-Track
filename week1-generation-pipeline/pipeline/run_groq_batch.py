import json
import os
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI, RateLimitError
from pydantic import ValidationError

from pipeline.example_selector import WordCountEmbedder, select_examples
from pipeline.generate import event_query, load_json, safe_event
from pipeline.recover import human_review_fallback, parse_model_json
from pipeline.schema import ExceptionCommunication
from pipeline.semantic import find_problems

MODEL = "openai/gpt-oss-20b"
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


def parse_draft(raw: str) -> ExceptionCommunication:
    return parse_model_json(raw)


def submit_groq(prompt: str):
    while True:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        prompt
                        + "\n\nReply with one JSON object only. Keys: exception_id, "
                        + "customer_facing_draft, internal_summary, grounded_commitments, "
                        + "citations, confidence, requires_human_review, review_reason. "
                        + "Each citation has source_id and clause_or_case. "
                        + "Do not put XML or parameter tags in any text field."
                    ),
                }],
                response_format={"type": "json_object"},
            )
            return response, response.choices[0].message.content
        except RateLimitError:
            print("rate limit, waiting 20 seconds")
            time.sleep(20)


def run_event(event, chosen_snippets, chosen_history, chosen_examples):
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
        response, raw = submit_groq(attempt_prompt)
        try:
            draft = parse_draft(raw)
            problems = find_problems(draft, allowed_ids)
        except ValidationError as error:
            problems = [str(error)]
        if not problems:
            return draft, attempt
        attempt_prompt = (
            prompt
            + "\n\nYour previous output failed these checks:\n- "
            + "\n- ".join(problems)
            + "\nSubmit a corrected JSON object."
        )

    raw_output = "" if draft is None else draft.model_dump_json()
    return human_review_fallback(event["exception_id"], raw_output), 3


def main():
    events = load_json("fixtures/exception_events.json")
    snippets = load_json("fixtures/sla_snippets.json")
    history = load_json("fixtures/resolution_history.json")
    bank = load_json("fixtures/few_shot_bank.json")
    snippet_by_id = {item["source_id"]: item for item in snippets}
    history_by_id = {item["case_id"]: item for item in history}
    embedder = WordCountEmbedder([example["input"] for example in bank])
    claude = json.loads(Path("outputs/claude_results.json").read_text(encoding="utf-8"))

    saved = []
    for event in events:
        chosen_snippets = [snippet_by_id[sid] for sid in event["snippet_ids"]]
        chosen_history = [history_by_id[cid] for cid in event["history_ids"]]
        chosen_examples = select_examples(event_query(event), bank, embedder.embed_texts, k=3)
        draft, attempts = run_event(event, chosen_snippets, chosen_history, chosen_examples)
        saved.append(draft.model_dump())
        print(event["exception_id"], "attempts", attempts, "review", draft.requires_human_review)
        time.sleep(20)

    Path("outputs/groq_results.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print("---")
    for event, claude_draft, groq_draft in zip(events, claude, saved):
        claude_review = claude_draft["requires_human_review"]
        groq_review = groq_draft["requires_human_review"]
        same = "same" if claude_review == groq_review else "DIFFERENT"
        print(
            event["exception_id"],
            "answer key",
            event["expects_human_review"],
            "claude",
            claude_review,
            "groq",
            groq_review,
            same,
        )


if __name__ == "__main__":
    main()