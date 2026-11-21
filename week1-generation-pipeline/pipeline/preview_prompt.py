import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

events = json.loads(Path("fixtures/exception_events.json").read_text(encoding="utf-8"))
snippets = json.loads(Path("fixtures/sla_snippets.json").read_text(encoding="utf-8"))
history = json.loads(Path("fixtures/resolution_history.json").read_text(encoding="utf-8"))
bank = json.loads(Path("fixtures/few_shot_bank.json").read_text(encoding="utf-8"))

event = events[0]
snippet_by_id = {item["source_id"]: item for item in snippets}
history_by_id = {item["case_id"]: item for item in history}

safe_event = {
    "exception_id": event["exception_id"],
    "status_code": event["status_code"],
    "carrier": event["carrier"],
    "country": event["country"],
    "commodity": event["commodity"],
    "hours_in_status": event["hours_in_status"],
    "freeform_note": event["freeform_note"],
}

env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("communication_prompt.jinja2")
prompt = template.render(
    current_event=safe_event,
    sla_snippets=[snippet_by_id[sid] for sid in event["snippet_ids"]],
    resolution_history=[history_by_id[cid] for cid in event["history_ids"]],
    few_shot_examples=bank[:3],
)

print(prompt)
print("---")
print("answer key leaked:", "expects_human_review" in prompt)