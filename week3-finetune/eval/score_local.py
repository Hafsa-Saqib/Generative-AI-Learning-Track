"""Score the untuned SmolLM2 model and the local LoRA adapter on the test split.

Both runs stay on this PC. A field is correct only when the parsed JSON matches
the rubric label. An unreadable reply counts as a miss.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_PATH = ROOT / "data" / "labeled_test.jsonl"
BASELINE_PATH = ROOT / "outputs" / "local_baseline_results.json"
TUNED_PATH = ROOT / "outputs" / "local_finetuned_results.json"

MISS = {
    "exception_subtype": "unparsed",
    "blocking_reason": "unparsed",
    "estimated_resolution_date": "unparsed",
    "requires_cold_chain_monitoring": "unparsed",
}


def _require_bool(value):
    if isinstance(value, bool):
        return value
    raise ValueError(value)


def _field(raw: dict, key: str, validator, miss: str):
    if not isinstance(raw, dict) or key not in raw:
        return miss, False
    value = raw[key]
    if key != "requires_cold_chain_monitoring" and value in ("null", "none", "None", ""):
        value = None
    if key == "requires_cold_chain_monitoring" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered not in ("true", "false"):
            return miss, False
        value = lowered == "true"
    try:
        return validator(value), True
    except (TypeError, ValueError):
        return miss, False


def parse_prediction(text: str) -> tuple[dict, bool]:
    from schema import ExtractionResult

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return dict(MISS), False
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return dict(MISS), False
    subtype, subtype_ok = _field(raw, "exception_subtype", ExtractionResult.known_subtype, "unparsed")
    reason, reason_ok = _field(raw, "blocking_reason", ExtractionResult.known_reason, "unparsed")
    date, date_ok = _field(raw, "estimated_resolution_date", ExtractionResult.known_date, "unparsed")
    cold, cold_ok = _field(raw, "requires_cold_chain_monitoring", _require_bool, "unparsed")
    return {
        "exception_subtype": subtype,
        "blocking_reason": reason,
        "estimated_resolution_date": date,
        "requires_cold_chain_monitoring": cold,
    }, subtype_ok and reason_ok and date_ok and cold_ok


def score_split(model_name: str, predict) -> dict:
    from eval.metrics import field_accuracy, percentile

    rows = []
    latencies = []
    parsed = 0
    items = [json.loads(line) for line in TEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, item in enumerate(items, start=1):
        started = time.perf_counter()
        text = predict(item["note"])
        latencies.append((time.perf_counter() - started) * 1000)
        prediction, ok = parse_prediction(text)
        parsed += int(ok)
        rows.append({**item, "prediction": prediction, "raw_reply": text, "latency_ms": latencies[-1]})
        print(f"{model_name} {index}/{len(items)} parsed={ok}", flush=True)
    ordered = sorted(latencies)
    return {
        "model": model_name,
        "rows": rows,
        "parsed": parsed,
        "accuracy": field_accuracy(rows),
        "latency_ms": {"p50": percentile(ordered, 50), "p95": percentile(ordered, 95)},
    }


def _predict_fn(model, tokenizer):
    import torch

    def predict(note: str) -> str:
        from finetune.format_for_fireworks import SYSTEM

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": note},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=192)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output[0][encoded["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return predict


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finetune.train_local import ADAPTER_DIR, BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    base.eval()
    baseline = score_split(BASE_MODEL, _predict_fn(base, tokenizer))
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tuned_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    tuned_model = PeftModel.from_pretrained(tuned_model, ADAPTER_DIR)
    tuned_model.eval()
    tuned = score_split(f"{BASE_MODEL}+lora", _predict_fn(tuned_model, tokenizer))

    BASELINE_PATH.parent.mkdir(exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    TUNED_PATH.write_text(json.dumps(tuned, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline": baseline["accuracy"],
        "finetuned": tuned["accuracy"],
        "baseline_parsed": baseline["parsed"],
        "finetuned_parsed": tuned["parsed"],
    }, indent=2))


if __name__ == "__main__":
    main()
