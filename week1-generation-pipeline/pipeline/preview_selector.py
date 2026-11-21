import json
from pathlib import Path
from pipeline.example_selector import WordCountEmbedder, cosine_similarity, select_examples

events = json.loads(Path("fixtures/exception_events.json").read_text(encoding="utf-8"))
bank = json.loads(Path("fixtures/few_shot_bank.json").read_text(encoding="utf-8"))
event = events[0]
query = (
    f"{event['status_code']} | {event['carrier']} | {event['country']} | "
    f"{event['commodity']} | {event['hours_in_status']} hours | {event['freeform_note']}"
)

embedder = WordCountEmbedder([example["input"] for example in bank])
chosen = select_examples(query, bank, embedder.embed_texts, k=3)
vectors = embedder.embed_texts([example["input"] for example in chosen])

print(query)
print("---")
for example in chosen:
    print(example["example_id"], example["input"][:90])

for left in range(len(chosen)):
    for right in range(left + 1, len(chosen)):
        score = cosine_similarity(vectors[left], vectors[right])
        print(chosen[left]["example_id"], chosen[right]["example_id"], round(score, 3))