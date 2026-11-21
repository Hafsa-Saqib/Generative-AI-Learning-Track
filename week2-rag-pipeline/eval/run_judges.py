"""Score the saved search results with RAGAS and DeepEval faithfulness.

Both judges use Claude. The script saves after every question, so a stopped
run continues where it left off. Set DEEPEVAL telemetry off before that import.
"""

import asyncio
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"

# This RAGAS release imports a Vertex chat class that the installed
# langchain-community package no longer ships. The judge uses Claude, so a
# stand-in class is enough to let the import finish.
_VERTEX_CHAT = "langchain_community.chat_models.vertexai"
if _VERTEX_CHAT not in sys.modules:
    import langchain_community.chat_models  # noqa: F401

    _vertex_chat = types.ModuleType(_VERTEX_CHAT)

    class ChatVertexAI:
        pass

    _vertex_chat.ChatVertexAI = ChatVertexAI
    sys.modules[_VERTEX_CHAT] = _vertex_chat

from anthropic import AsyncAnthropic
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase
from ragas.llms.base import llm_factory
from ragas.metrics.collections import Faithfulness
import instructor.function_calls as instructor_calls

_original_parse = instructor_calls.OpenAISchema.parse_anthropic_json.__func__


@classmethod
def _parse_anthropic_json(cls, completion, validation_context=None, strict=None):
    """Sonnet 5 can lead with a thinking block. The score is in the text block."""
    text_blocks = [block for block in getattr(completion, "content", []) if hasattr(block, "text")]
    if text_blocks and completion.content[0] is not text_blocks[0]:
        completion = completion.model_copy(update={"content": text_blocks})
    return _original_parse(cls, completion, validation_context, strict)


instructor_calls.OpenAISchema.parse_anthropic_json = _parse_anthropic_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "search_results.json"
OUT = ROOT / "outputs" / "judge_results.json"
MODEL = "claude-sonnet-5"


def load_done() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"questions": {}}


def save(payload: dict) -> None:
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def is_rate_limit(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "toomanyrequests" in name:
        return True
    return getattr(exc, "status_code", None) == 429


def ragas_score(metric: Faithfulness, question: dict) -> float:
    result = asyncio.run(metric.ascore(
        user_input=question["question"],
        response=question["draft"]["customer_facing_draft"],
        retrieved_contexts=[chunk["text"] for chunk in question["top"]],
    ))
    return float(result.value)


def deepeval_score(metric: FaithfulnessMetric, question: dict) -> float:
    case = LLMTestCase(
        input=question["question"],
        actual_output=question["draft"]["customer_facing_draft"],
        retrieval_context=[chunk["text"] for chunk in question["top"]],
    )
    metric.measure(case)
    return float(metric.score)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in this terminal first.")
    questions = json.loads(RESULTS.read_text(encoding="utf-8"))["questions"]
    done = load_done()
    ragas_llm = llm_factory(
        MODEL,
        provider="anthropic",
        client=AsyncAnthropic(),
    )
    # Claude Sonnet 5 rejects sampling controls.
    ragas_llm.model_args.pop("temperature", None)
    ragas_llm.model_args.pop("top_p", None)
    ragas_metric = Faithfulness(llm=ragas_llm)
    deepeval_metric = FaithfulnessMetric(
        model=AnthropicModel(model=MODEL),
        include_reason=False,
        async_mode=False,
    )
    for index, question in enumerate(questions, start=1):
        row = done["questions"].setdefault(question["id"], {})
        if "ragas_faithfulness" in row and "deepeval_faithfulness" in row:
            print(f"{index}/{len(questions)} {question['id']} already scored", flush=True)
            continue
        print(f"{index}/{len(questions)} {question['id']}", flush=True)
        for attempt in range(4):
            try:
                if "ragas_faithfulness" not in row:
                    row["ragas_faithfulness"] = ragas_score(ragas_metric, question)
                if "deepeval_faithfulness" not in row:
                    row["deepeval_faithfulness"] = deepeval_score(deepeval_metric, question)
                break
            except Exception as exc:
                limited = is_rate_limit(exc)
                if attempt == 3 or not limited:
                    raise
                print(f"Waiting 30 seconds after {type(exc).__name__}.", flush=True)
                time.sleep(30)
        save(done)
        time.sleep(1)
    ragas_values = [row["ragas_faithfulness"] for row in done["questions"].values() if row.get("ragas_faithfulness") == row.get("ragas_faithfulness")]
    deepeval_values = [row["deepeval_faithfulness"] for row in done["questions"].values()]
    ragas_values = [value for value in ragas_values if value is not None and value == value]
    deepeval_values = [value for value in deepeval_values if value is not None and value == value]
    done["metrics"] = {
        "questions": len(questions),
        "ragas_faithfulness": mean(ragas_values),
        "deepeval_faithfulness": mean(deepeval_values),
        "judge_model": MODEL,
    }
    save(done)
    print(json.dumps(done["metrics"], indent=2))


if __name__ == "__main__":
    main()
