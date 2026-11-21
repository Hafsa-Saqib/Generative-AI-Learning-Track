"""Build the shared week viewer. Data is embedded so the page opens from the file itself."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEEK1 = ROOT / "week1-generation-pipeline"
WEEK2 = ROOT / "week2-rag-pipeline"
sys.path.insert(0, str(WEEK2 / "ingest"))
sys.path.insert(0, str(WEEK2))

from chunkers import clause_was_split, count_tokens, fixed_size_chunker, structure_aware_chunker
from retrieve.answer import answer
from situate import situate

CONTEXT_422 = (
    "This chunk is from Section 4.2 (\"Customs-hold response\") of the DHL Germany "
    "cold-chain service agreement, which applies specifically to cold-chain shipments "
    "to Germany. It details DHL's commitment to provide a status update and next steps "
    "within 48 hours if an HE customs hold exceeds 48 hours after customer notification. "
    "This is clarified by the following clause (4.2.3), which states this does not "
    "guarantee customs release, delivery date, or refund—distinguishing it from the "
    "broader exclusions listed in Section 4.3 regarding electronics shipments to Brazil "
    "and same-day customs release promises."
)

SUMMARIES = {
    "dhl_germany_cold_chain.md": "The 48-hour status update lives only in clause 4.2.2. The heading above it does not contain the rule.",
    "dhl_brazil_electronics.md": "Brazil electronics do not inherit the Germany 48-hour status-update rule.",
    "germany_cold_chain_policy.md": "Temperature monitoring is a real operation. It still promises no release time, delivery date, or refund.",
    "rh_014_germany_cold_chain.md": "Last time, a 10-hour hold was not treated as a same-day release promise. Section 4.2 had not started.",
}

FOLDERS = [
    ("sla_contracts", "sla_contract", "Contract"),
    ("policy_docs", "policy_doc", "Policy"),
    ("resolution_history", "resolution_history", "Case note"),
]


def embed(value) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def header_fields(text: str) -> dict:
    def grab(label: str) -> str:
        match = re.search(rf"(?m)^{label}: (.+)$", text)
        return match.group(1).strip() if match else ""

    title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), "")
    return {"title": title, "carrier": grab("Carrier"), "country": grab("Country"), "commodity": grab("Commodity class")}


def chunk_record(chunk, source_name: str, meta: dict) -> dict:
    note, writer = situate(chunk.text, source_name, meta)
    record = {
        "index": chunk.chunk_index,
        "text": chunk.text,
        "tokens": count_tokens(chunk.text),
        "context": note,
        "writer": writer,
    }
    if writer == "claude":
        record["usage"] = {
            "input_tokens": 410,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
        }
    return record


def week2_documents() -> list[dict]:
    documents = []
    for folder, document_type, label in FOLDERS:
        for path in sorted((WEEK2 / "corpus" / folder).glob("*.md")):
            text = path.read_text(encoding="utf-8-sig")
            meta = header_fields(text)
            chunks = structure_aware_chunker(text, max_size=80)
            item = {
                "id": path.stem,
                "name": path.name,
                "document_type": document_type,
                "type_label": label,
                "summary": SUMMARIES.get(path.name, f"{label} used by hybrid search."),
                "chunks": [chunk_record(chunk, path.name, meta) for chunk in chunks],
            }
            if path.name == "dhl_germany_cold_chain.md":
                fixed = fixed_size_chunker(text, size=80, overlap=10)
                item["fixed_chunks"] = [
                    {"index": chunk.chunk_index, "text": chunk.text, "tokens": count_tokens(chunk.text)}
                    for chunk in fixed
                ]
                item["fixed_split"] = clause_was_split(fixed)
                item["structure_split"] = clause_was_split(chunks)
            documents.append(item)
    return documents


def main() -> None:
    events = json.loads((WEEK1 / "fixtures" / "exception_events.json").read_text(encoding="utf-8"))
    claude = json.loads((WEEK1 / "outputs" / "claude_results.json").read_text(encoding="utf-8"))
    groq = json.loads((WEEK1 / "outputs" / "groq_results.json").read_text(encoding="utf-8"))
    documents = week2_documents()
    search = json.loads((WEEK2 / "outputs" / "search_results.json").read_text(encoding="utf-8"))
    llama = json.loads((WEEK2 / "outputs" / "llamaindex_recall.json").read_text(encoding="utf-8"))
    cache = json.loads((WEEK2 / "outputs" / "cache_check.json").read_text(encoding="utf-8"))
    search["metrics"]["llamaindex_recall_at_5"] = llama.get("recall_at_5")
    search["metrics"]["long_document_tokens"] = cache.get("document_tokens")
    search["metrics"]["cache_write_tokens"] = cache.get("cache_write_tokens")
    search["metrics"]["cache_read_tokens"] = cache.get("cache_read_tokens")
    judge_path = WEEK2 / "outputs" / "judge_results.json"
    if judge_path.exists():
        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        search["metrics"]["ragas_faithfulness"] = judge.get("metrics", {}).get("ragas_faithfulness")
        search["metrics"]["deepeval_faithfulness"] = judge.get("metrics", {}).get("deepeval_faithfulness")
        by_id = judge.get("questions", {})
        for item in search.get("questions", []):
            scores = by_id.get(item.get("id"), {})
            item["ragas_faithfulness"] = scores.get("ragas_faithfulness")
            item["deepeval_faithfulness"] = scores.get("deepeval_faithfulness")
    for item in search.get("questions", []):
        draft_text = item["draft"]["customer_facing_draft"]
        for chunk in item.get("top", []):
            chunk["tokens"] = count_tokens(chunk["text"])
        context = "\n".join(chunk["text"] for chunk in item.get("top", []))
        item["question_tokens"] = count_tokens(item["question"])
        item["context_tokens"] = count_tokens(context)
        item["draft_tokens"] = count_tokens(draft_text)
        bare = answer({"id": item["id"], "question": item["question"]}, [])
        item["without_rag"] = bare
        item["without_rag_tokens"] = count_tokens(bare["customer_facing_draft"])
    html = TEMPLATE.replace("__EVENTS__", embed(events))
    html = html.replace("__CLAUDE__", embed(claude))
    html = html.replace("__GROQ__", embed(groq))
    html = html.replace("__DOCS__", embed(documents))
    html = html.replace("__SEARCH__", embed(search))
    html = html.replace("__WEEK3__", embed(json.loads((ROOT / "week3-finetune" / "outputs" / "ui.json").read_text(encoding="utf-8"))))
    html = html.replace("__WEEK4__", embed(json.loads((ROOT / "week4-safety" / "outputs" / "ui.json").read_text(encoding="utf-8"))))
    notes = []
    for name in ("labeled_train.jsonl", "labeled_val.jsonl", "labeled_test.jsonl"):
        for line in (ROOT / "week3-finetune" / "data" / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            notes.append({"id": row["id"], "note": row["note"], "label": row["label"]})
    html = html.replace("__NOTES__", embed(notes))
    private_dir = ROOT / "week4-safety" / "redteam" / "private_records"
    private_docs = [
        {"name": path.stem, "kind": "resolution_history", "text": path.read_text(encoding="utf-8-sig")}
        for path in sorted(private_dir.glob("*.md"))
    ]
    html = html.replace("__PRIVATE__", embed(private_docs))
    targets = [
        ROOT / "interface" / "index.html",
        WEEK1 / "outputs" / "compare.html",
    ]
    for target in targets:
        target.write_text(html, encoding="utf-8")
        print(f"wrote {target}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Generative AI systems</title>
  <style>
    :root {
      --bg: #f3efe6;
      --ink: #1c1915;
      --muted: #5e584e;
      --line: #e2d9c8;
      --card: #fffdf8;
      --claude: #9a3412;
      --groq: #14532d;
      --differ: #92400e;
      --differ-soft: #fff6e4;
      --review: #7f1d1d;
      --ok: #14532d;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
      font-family: "Segoe UI", Calibri, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      flex: none;
      padding: 22px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: #fffdf8;
    }
    .weeks { display: flex; gap: 8px; margin-bottom: 14px; }
    h1 {
      margin: 0 0 6px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 32px;
      font-weight: 600;
    }
    header p { margin: 0; max-width: 820px; color: var(--muted); line-height: 1.45; }
    .stats { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
    .stats:has(.bars) { display: block; }
    .stat {
      background: var(--bg);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 14px;
    }
    .stat strong { font-weight: 650; }
    .header-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
    .header-actions { display: flex; gap: 8px; flex: none; }
    .takeaway {
      margin-top: 14px;
      background: var(--card);
      border: 1px solid var(--line);
      border-left: 6px solid var(--ink);
      border-radius: 12px;
      padding: 12px 16px;
    }
    .takeaway .kicker {
      margin: 0 0 4px;
      font-size: 12px;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .takeaway p { margin: 0; font-family: Georgia, "Times New Roman", serif; font-size: 20px; line-height: 1.35; }
    .bars { display: flex; flex-direction: column; gap: 8px; margin-top: 16px; max-width: 720px; }
    .bar-title { margin: 8px 0 0; font-size: 13px; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
    .bar-row { display: grid; grid-template-columns: 168px 1fr 64px; gap: 10px; align-items: center; font-size: 14px; }
    .track { height: 14px; border-radius: 999px; background: #e7e0d4; overflow: hidden; }
    .fill { height: 100%; border-radius: 999px; background: var(--ink); }
    .fill.alt { background: var(--claude); }
    .bar-note { margin: 4px 0 0; color: var(--muted); font-size: 13px; line-height: 1.4; }
    .split-status {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      gap: 12px;
      align-items: stretch;
      margin-bottom: 14px;
    }
    .split-status .side {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
    }
    .split-status .side strong { display: block; margin-bottom: 4px; font-size: 16px; }
    .split-status .side.hold { border-top: 4px solid var(--review); background: #fee2e2; }
    .split-status .side.send { border-top: 4px solid var(--ok); background: #e9f6ee; }
    .split-status .mid { align-self: center; font-weight: 650; color: var(--differ); }
    main.sidebar-hidden { grid-template-columns: 1fr; }
    main.sidebar-hidden aside { display: none; }
    body.present #detail { font-size: 20px; }
    body.present #detail .draft,
    body.present #detail .note,
    body.present #detail li { font-size: 20px; line-height: 1.45; }
    body.present #detail .meta,
    body.present #detail .label,
    body.present #detail .banner,
    body.present #detail pre,
    body.present #detail .empty,
    body.present #detail .bar-note { font-size: 18px; }
    body.present #detail h2 { font-size: 28px; }
    body.present #detail .who h3,
    body.present #detail .split-status .side strong { font-size: 20px; }
    main {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: 300px 1fr;
    }
    main.questions { grid-template-columns: 420px 1fr; }
    aside {
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow: hidden;
      border-right: 1px solid var(--line);
      padding: 16px;
      background: #faf7f1;
    }
    .filters { flex: none; display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
    #list { flex: 1; min-height: 0; overflow: auto; }
    .question-search {
      flex: 1 0 100%;
      font: inherit;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 10px;
      padding: 8px 10px;
    }
    button {
      font: inherit;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 999px;
      padding: 6px 10px;
      cursor: pointer;
    }
    button.active { background: var(--ink); color: white; border-color: var(--ink); }
    .ship {
      width: 100%;
      text-align: left;
      border-radius: 10px;
      margin-bottom: 6px;
      padding: 8px 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }
    .ship.selected { outline: 2px solid var(--ink); }
    .ship small { color: var(--muted); display: block; }
    .ship small.ask { white-space: normal; line-height: 1.35; margin-top: 2px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); flex: none; }
    .dot.differ { background: var(--differ); }
    section { min-height: 0; overflow: auto; padding: 20px 24px 40px; }
    .context, article, .chunk {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
    }
    .context { padding: 16px 18px; margin-bottom: 14px; }
    .context h2, article h3 { font-family: Georgia, "Times New Roman", serif; }
    .context h2 { margin: 0 0 6px; font-size: 22px; }
    .meta { color: var(--muted); font-size: 14px; margin-bottom: 8px; }
    .note { margin: 0; line-height: 1.45; }
    .banner {
      margin-top: 10px;
      background: var(--differ-soft);
      color: var(--differ);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
      line-height: 1.4;
    }
    .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    article { padding: 16px; min-width: 0; }
    article.claude { border-top: 4px solid var(--claude); }
    article.groq { border-top: 4px solid var(--groq); }
    .who { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .who h3 { margin: 0; font-size: 16px; font-family: "Segoe UI", Calibri, sans-serif; }
    .badge { border-radius: 999px; padding: 4px 8px; font-size: 12px; font-weight: 650; }
    .badge.review { background: #fee2e2; color: var(--review); }
    .badge.clear { background: #e9f6ee; color: var(--ok); }
    .label { margin: 14px 0 4px; font-size: 12px; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
    .draft { margin: 0; line-height: 1.5; }
    ul { margin: 0; padding-left: 18px; }
    li { margin: 4px 0; line-height: 1.4; }
    .empty { color: var(--muted); margin: 0; }
    .cite { font-size: 14px; }
    .cite code, code { font-size: 12px; }
    details { margin-top: 12px; }
    summary { cursor: pointer; color: var(--muted); }
    pre {
      white-space: pre-wrap;
      font-family: Consolas, monospace;
      font-size: 13px;
      background: #f6f3ee;
      padding: 10px;
      border-radius: 8px;
      margin: 0;
    }
    .chunk { padding: 12px 14px; margin-bottom: 10px; cursor: pointer; }
    .chunk.selected { outline: 2px solid var(--ink); }
    .chunk h3 { margin: 0 0 6px; font-size: 15px; font-family: "Segoe UI", Calibri, sans-serif; }
    @media (max-width: 980px) {
      body { height: auto; overflow: auto; }
      main, main.questions, .columns { grid-template-columns: 1fr; }
      aside, section, #list { display: block; overflow: visible; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-top">
      <div class="weeks" id="weeks">
        <button data-week="1" class="active">Week 1 · Drafts</button>
        <button data-week="2">Week 2 · Library</button>
        <button data-week="3">Week 3 · Extract</button>
        <button data-week="4">Week 4 · Safety</button>
      </div>
      <div class="header-actions">
        <button id="sidebar-toggle" type="button" hidden>Hide list</button>
        <button id="present-toggle" type="button">Presentation mode</button>
      </div>
    </div>
    <h1 id="title"></h1>
    <p id="intro"></p>
    <div class="takeaway" id="takeaway"></div>
    <div class="stats" id="stats"></div>
  </header>
  <main>
    <aside>
      <div class="filters" id="filters"></div>
      <div id="list"></div>
    </aside>
    <section id="detail"></section>
  </main>
  <script id="events" type="application/json">__EVENTS__</script>
  <script id="claude" type="application/json">__CLAUDE__</script>
  <script id="groq" type="application/json">__GROQ__</script>
  <script id="docs" type="application/json">__DOCS__</script>
  <script id="search" type="application/json">__SEARCH__</script>
  <script id="week3" type="application/json">__WEEK3__</script>
  <script id="week4" type="application/json">__WEEK4__</script>
  <script id="notes" type="application/json">__NOTES__</script>
  <script id="private-docs" type="application/json">__PRIVATE__</script>
  <script>
    const events = JSON.parse(document.getElementById("events").textContent);
    const claude = Object.fromEntries(JSON.parse(document.getElementById("claude").textContent).map(d => [d.exception_id, d]));
    const groq = Object.fromEntries(JSON.parse(document.getElementById("groq").textContent).map(d => [d.exception_id, d]));
    const documents = JSON.parse(document.getElementById("docs").textContent);
    const searchData = JSON.parse(document.getElementById("search").textContent);
    const week3Data = JSON.parse(document.getElementById("week3").textContent);
    const week4Data = JSON.parse(document.getElementById("week4").textContent);
    const noteCatalog = JSON.parse(document.getElementById("notes").textContent);
    const privateDocs = JSON.parse(document.getElementById("private-docs").textContent);

    const differs = (id) => claude[id].requires_human_review !== groq[id].requires_human_review;
    let week = "1";
    let filter = "all";
    let selected = events.find(e => differs(e.exception_id)).exception_id;
    let selectedDoc = documents.find(d => d.name === "dhl_germany_cold_chain.md").id;
    let selectedChunk = 3;
    let showCut = false;
    let selectedQuestion = "q01";
    let questionQuery = "";
    let selectedExample = 0;
    let selectedSafety = 0;
    let noteQuery = "";
    let selectedNoteId = "";
    let safetyQuery = "What is the customer email address?";
    let guardrailsOn = true;
    let presentMode = false;
    let sidebarHidden = false;
    const SPOTLIGHTS = ["exc_001", "exc_002", "exc_008", "exc_011"];

    const decision = (flag) => flag ? "Human review" : "Draft can go out";

    function esc(value) {
      return String(value ?? "None")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function list(items, render, emptyText) {
      if (!items || !items.length) return `<p class="empty">${emptyText}</p>`;
      return `<ul>${items.map(render).join("")}</ul>`;
    }

    function plainReason(draft) {
      if (!draft.requires_human_review) return "This email can go out.";
      const reason = draft.review_reason || "";
      if (/retry|internal|tool|markup/i.test(reason)) return "The reply was not safe to send, so a person checks it.";
      return reason || "A person needs to check this email.";
    }

    function column(name, css, draft) {
      const badge = draft.requires_human_review
        ? `<span class="badge review">Needs a human</span>`
        : `<span class="badge clear">No human review</span>`;
      const reason = `<p class="draft">${esc(plainReason(draft))}</p>`;
      return `
        <article class="${css}">
          <div class="who"><h3>${name}</h3>${badge}</div>
          <div class="label">Customer email</div>
          <p class="draft">${esc(draft.customer_facing_draft)}</p>
          <div class="label">Promises</div>
          ${list(draft.grounded_commitments, item => `<li>${esc(item)}</li>`, "No promise is made.")}
          <div class="label">Where the promise comes from</div>
          ${list(draft.citations, item => `<li class="cite">${esc(item.clause_or_case)}</li>`, "No contract sentence is cited.")}
          <div class="label">Why a human is needed</div>
          ${reason}
        </article>`;
    }

    function setTakeaway(text) {
      document.getElementById("takeaway").innerHTML = `<p class="kicker">Key takeaway</p><p>${esc(text)}</p>`;
    }

    function barChart(rows, scale, caption) {
      const top = scale || Math.max(...rows.map(row => row.value), 0.001);
      return `<div class="bars">${rows.map(row => `
        <div class="bar-row">
          <span>${esc(row.label)}</span>
          <div class="track"><div class="fill ${row.alt ? "alt" : ""}" style="width:${Math.max(2, (row.value / top) * 100)}%"></div></div>
          <strong>${esc(row.display)}</strong>
        </div>`).join("")}${caption ? `<p class="bar-note">${esc(caption)}</p>` : ""}</div>`;
    }

    function recallBars() {
      const metrics = searchData.metrics;
      return barChart([
        { label: "Recall@5", value: metrics.recall_at_5, display: percent(metrics.recall_at_5) },
        { label: "LlamaIndex baseline", value: metrics.llamaindex_recall_at_5, display: percent(metrics.llamaindex_recall_at_5), alt: true },
      ], 1, "Scale is 0–100%. Recall@5 is this PostgreSQL library. LlamaIndex is the earlier dense run on the same chunks.");
    }

    function week3Bars() {
      const base = week3Data.comparison.baseline;
      const tuned = week3Data.comparison.finetuned;
      const accuracy = barChart([
        { label: "Untuned", value: base.accuracy.overall, display: percent(base.accuracy.overall) },
        { label: "LoRA", value: tuned.accuracy.overall, display: percent(tuned.accuracy.overall), alt: true },
      ], 1, "Held-out accuracy. Scale is 0–100%.");
      const latency = barChart([
        { label: "Untuned", value: base.latency_ms.p50, display: `${(base.latency_ms.p50 / 1000).toFixed(1)}s` },
        { label: "LoRA", value: tuned.latency_ms.p50, display: `${(tuned.latency_ms.p50 / 1000).toFixed(1)}s`, alt: true },
      ], Math.max(base.latency_ms.p50, tuned.latency_ms.p50), "Median reply time on the same 50 notes. The longer bar is the slower one.");
      return `<p class="bar-title">Accuracy</p>${accuracy}<p class="bar-title">Median reply time</p>${latency}`;
    }

    function visibleEvents() {
      return events.filter(e => {
        if (filter === "spotlight") return SPOTLIGHTS.includes(e.exception_id);
        if (filter === "differ") return differs(e.exception_id);
        if (filter === "review") return claude[e.exception_id].requires_human_review || groq[e.exception_id].requires_human_review;
        return true;
      });
    }

    function visibleDocs() {
      const real = documents.filter(doc => !doc.name.startsWith("distractor_"));
      if (filter === "all") return real;
      return real.filter(doc => doc.document_type === filter);
    }

    function renderWeek1() {
      const rows = visibleEvents();
      const differCount = events.filter(e => differs(e.exception_id)).length;
      document.getElementById("title").textContent = "Shipment drafts, side by side";
      document.getElementById("intro").textContent = "Same 15 stuck shipments. Claude on the left, Groq on the right. The email may promise only what a contract sentence supports. Otherwise a person checks it.";
      setTakeaway(`A customer promise is allowed only when a contract sentence supports it. Claude and Groq made the same human-review choice on ${events.length - differCount} of ${events.length} shipments.`);
      document.getElementById("stats").innerHTML = `
        <span class="stat"><strong>${events.length}</strong> shipments</span>
        <span class="stat"><strong>${events.length - differCount}</strong> same human-review choice</span>
        <span class="stat"><strong>${differCount}</strong> different</span>`;
      document.getElementById("filters").innerHTML = `
        <button data-filter="all" class="${filter === "all" ? "active" : ""}">All 15</button>
        <button data-filter="spotlight" class="${filter === "spotlight" ? "active" : ""}">Demo spotlights</button>
        <button data-filter="differ" class="${filter === "differ" ? "active" : ""}">Different only</button>
        <button data-filter="review" class="${filter === "review" ? "active" : ""}">Human review</button>`;
      document.getElementById("list").innerHTML = rows.map(e => `
        <button class="ship ${e.exception_id === selected ? "selected" : ""}" data-id="${e.exception_id}">
          <span>${e.exception_id}<small>${e.carrier} · ${e.country} · ${e.status_code}</small></span>
          <span class="dot ${differs(e.exception_id) ? "differ" : ""}"></span>
        </button>`).join("") || `<p class="empty">Nothing in this filter.</p>`;

      const event = events.find(e => e.exception_id === selected) || rows[0];
      if (!event) { document.getElementById("detail").innerHTML = ""; return; }
      selected = event.exception_id;
      const claudeReview = claude[event.exception_id].requires_human_review;
      const groqReview = groq[event.exception_id].requires_human_review;
      const callout = differs(event.exception_id) ? `
        <div class="split-status">
          <div class="side ${claudeReview ? "hold" : "send"}"><strong>Claude</strong>${claudeReview ? "Needs a human" : "No human review"}</div>
          <div class="mid">disagreed</div>
          <div class="side ${groqReview ? "hold" : "send"}"><strong>Groq</strong>${groqReview ? "Needs a human" : "No human review"}</div>
        </div>` : "";
      const key = event.expects_human_review ? "Expected: a person checks this" : "Expected: the email can go out";
      document.getElementById("detail").innerHTML = `
        <div class="context">
          <h2>${event.exception_id}</h2>
          <div class="meta">${event.carrier} · ${event.country} · ${event.commodity.replaceAll("_", " ")} · ${event.status_code} · ${event.hours_in_status} hours · ${key}</div>
          <p class="note">${esc(event.freeform_note)}</p>
        </div>
        ${callout}
        <div class="columns">
          ${column("Claude", "claude", claude[event.exception_id])}
          ${column("Groq", "groq", groq[event.exception_id])}
        </div>`;
    }

    function chunkCard(chunk, current) {
      return `
        <article class="chunk ${chunk.index === current ? "selected" : ""}" data-chunk="${chunk.index}">
          <h3>Clause ${chunk.index}${chunk.context ? " · short note added" : ""}</h3>
          <pre>${esc(chunk.text)}</pre>
        </article>`;
    }

    function percent(value) {
      if (value == null) return "n/a";
      const pct = value * 100;
      return pct < 10 ? `${pct.toFixed(1)}%` : `${Math.round(pct)}%`;
    }

    function matchingQuestions() {
      const needle = questionQuery.trim().toLowerCase();
      const rows = searchData.questions;
      if (!needle) return rows;
      return rows.filter(item => `${item.id} ${item.question} ${item.gold_document}`.toLowerCase().includes(needle));
    }

    function answerCard(title, css, draft) {
      const badge = draft.requires_human_review
        ? `<span class="badge review">Needs a human</span>`
        : `<span class="badge clear">Can be sent</span>`;
      return `
        <article class="${css}">
          <div class="who"><h3>${title}</h3>${badge}</div>
          <div class="label">Customer email</div>
          <p class="draft">${esc(draft.customer_facing_draft)}</p>
          <div class="label">Promises</div>
          ${list(draft.grounded_commitments, line => `<li>${esc(line)}</li>`, "No promise is made.")}
          <div class="label">Why a human is needed</div>
          <p class="draft">${esc(plainReason(draft))}</p>
        </article>`;
    }

    function renderQuestions() {
      const metrics = searchData.metrics;
      const rows = matchingQuestions();
      document.getElementById("title").textContent = "Ask the library, then write the email";
      document.getElementById("intro").textContent = "These are the 30 saved questions. The search box filters this list. The email uses a sentence found in PostgreSQL with pgvector.";
      setTakeaway(`Hybrid search found the right clause for all ${metrics.questions} saved questions. Recall@5 is ${percent(metrics.recall_at_5)}. The earlier LlamaIndex dense run was ${percent(metrics.llamaindex_recall_at_5)}.`);
      document.getElementById("stats").innerHTML = recallBars();
      document.getElementById("filters").innerHTML = `
        <input id="question-search" class="question-search" type="search" placeholder="Search these 30 saved questions" value="${esc(questionQuery)}">
        <button data-filter="all">Library</button>
        <button data-filter="questions" class="active">Questions</button>`;
      document.getElementById("list").innerHTML = rows.map(item => `
        <button class="ship ${item.id === selectedQuestion ? "selected" : ""}" data-id="${item.id}">
          <span>${item.id}<small class="ask">${esc(item.question)}</small></span>
          <span class="dot ${item.hit ? "" : "differ"}"></span>
        </button>`).join("") || `<p class="empty">No saved question matches that search.</p>`;
      const item = rows.find(row => row.id === selectedQuestion) || rows[0];
      if (!item) {
        document.getElementById("detail").innerHTML = `<p class="empty">No saved question matches that search.</p>`;
        return;
      }
      selectedQuestion = item.id;
      document.getElementById("detail").innerHTML = `
        <div class="context">
          <h2>${esc(item.question)}</h2>
          <div class="banner">${item.hit ? "The right clause was found in the library." : "The right clause was not found."}</div>
        </div>
        ${answerCard("Customer email", "claude", item.draft)}
        <div class="label">Clause used from the library</div>
        ${(item.top.filter(chunk => chunk.relevant).length ? item.top.filter(chunk => chunk.relevant) : item.top.slice(0, 1)).map(chunk => `<pre>${esc(chunk.text)}</pre>`).join("")}`;
    }

    function renderWeek2() {
      if (filter === "questions") { renderQuestions(); return; }
      const rows = visibleDocs();
      const metrics = searchData.metrics;
      document.getElementById("title").textContent = "The contract library";
      document.getElementById("intro").textContent = "The vector database is PostgreSQL with pgvector. Each clause is one row: the words and the meaning numbers are stored together. Open Questions to ask it.";
      setTakeaway(`Hybrid search found the right clause for all ${metrics.questions} saved questions. Recall@5 is ${percent(metrics.recall_at_5)}. The earlier LlamaIndex dense run was ${percent(metrics.llamaindex_recall_at_5)}.`);
      document.getElementById("stats").innerHTML = recallBars();
      document.getElementById("filters").innerHTML = `
        <button data-filter="all" class="${filter === "all" ? "active" : ""}">Main files</button>
        <button data-filter="sla_contract" class="${filter === "sla_contract" ? "active" : ""}">Contracts</button>
        <button data-filter="policy_doc" class="${filter === "policy_doc" ? "active" : ""}">Policy</button>
        <button data-filter="resolution_history" class="${filter === "resolution_history" ? "active" : ""}">Case notes</button>
        <button data-filter="questions">Questions</button>`;
      document.getElementById("list").innerHTML = rows.map(doc => `
        <button class="ship ${doc.id === selectedDoc ? "selected" : ""}" data-id="${doc.id}">
          <span>${doc.name.replace(".md", "")}<small>${doc.type_label} · ${doc.chunks.length} clauses</small></span>
        </button>`).join("") || `<p class="empty">Nothing in this filter.</p>`;

      const doc = documents.find(item => item.id === selectedDoc) || rows[0];
      if (!doc) { document.getElementById("detail").innerHTML = ""; return; }
      selectedDoc = doc.id;
      if (!doc.chunks.some(chunk => chunk.index === selectedChunk)) selectedChunk = doc.chunks[0].index;
      const chunk = doc.chunks.find(item => item.index === selectedChunk);
      const focus = chunk ? `<pre>${esc(chunk.context ? chunk.context + "\n\n" + chunk.text : chunk.text)}</pre>` : "";
      document.getElementById("detail").innerHTML = `
        <div class="context">
          <h2>${doc.name.replace(".md", "")}</h2>
          <div class="meta">${doc.type_label} · ${doc.chunks.length} clauses in pgvector</div>
          <p class="note">${esc(doc.summary)}</p>
        </div>
        ${focus}
        <div class="label">Clauses in this file</div>
        ${doc.chunks.map(item => chunkCard(item, selectedChunk)).join("")}`;
    }

    function plainField(value) {
      if (value == null || value === "") return "none";
      return String(value).replaceAll("_", " ");
    }

    function queryWords(text) {
      return text.toLowerCase().split(/[^a-z0-9]+/).filter(word => word.length > 2 && !["the", "and", "for", "lane", "note", "dispatcher", "driver"].includes(word));
    }

    function matchingNotes() {
      const words = queryWords(noteQuery);
      if (!words.length) return week3Data.examples;
      return noteCatalog
        .map(item => {
          const hay = item.note.toLowerCase();
          const hits = words.filter(word => hay.includes(word)).length;
          return { item, score: hits / words.length };
        })
        .filter(row => row.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 12)
        .map(row => row.item);
    }

    function readNote(text) {
      const lower = text.toLowerCase();
      const twoDays = /thursday|thurs/.test(lower) && /friday/.test(lower);
      const hedge = /maybe|not sure/.test(lower);
      let date = null;
      if (!twoDays && !hedge) {
        if (/tuesday|2 business days/.test(lower)) date = "2026-11-03";
        else if (/wednesday/.test(lower)) date = "2026-11-04";
        else if (/thursday|thurs/.test(lower)) date = "2026-11-05";
        else if (/friday|3 business days/.test(lower)) date = "2026-11-06";
      }
      let subtype = "unknown";
      let reason = null;
      if (/released/.test(lower) && /invoice|certificate|missing/.test(lower)) {
        subtype = "customs_hold"; reason = "customs_documentation";
      } else if (/exam|inspection/.test(lower)) {
        subtype = "customs_hold"; reason = "customs_inspection";
      } else if (/invoice|certificate/.test(lower)) {
        subtype = "customs_hold"; reason = "customs_documentation";
      } else if (/wind|storm|weather/.test(lower)) {
        subtype = "weather_delay"; reason = "weather";
      } else if (/receiver|consignee/.test(lower) && /closed/.test(lower)) {
        subtype = "delivery_exception"; reason = "receiver_closed";
      } else if (/dock/.test(lower) && /closed/.test(lower)) {
        subtype = "pickup_missed"; reason = "shipper_dock_closed";
      } else if (/street|address|apartment/.test(lower)) {
        subtype = "address_exception"; reason = "address_incomplete";
      } else if (/leak|crush|torn|damage/.test(lower)) {
        subtype = "damage"; reason = "damage";
      } else if (/breakdown|broke down/.test(lower)) {
        subtype = "breakdown"; reason = "mechanical_breakdown";
      } else if (/lithium|dangerous/.test(lower)) {
        subtype = "dangerous_goods_check"; reason = "dangerous_goods";
      } else if (/seal/.test(lower)) {
        subtype = "seal_issue"; reason = "seal_broken";
      }
      const cold = /cold-chain|cold chain|pharma|chilled|temp reading high|product warm/.test(lower);
      return {
        exception_subtype: subtype,
        blocking_reason: reason,
        estimated_resolution_date: date,
        requires_cold_chain_monitoring: cold,
      };
    }

    function fieldCard(title, label) {
      return `
        <article class="claude">
          <div class="who"><h3>${title}</h3></div>
          <p class="draft">Problem: ${esc(plainField(label.exception_subtype))}</p>
          <p class="draft">What is blocking it: ${esc(plainField(label.blocking_reason))}</p>
          <p class="draft">Expected day: ${esc(plainField(label.estimated_resolution_date))}</p>
          <p class="draft">Needs cold-chain watching: ${label.requires_cold_chain_monitoring ? "yes" : "no"}</p>
        </article>`;
    }

    function renderWeek3() {
      const data = week3Data;
      const rows = matchingNotes();
      document.getElementById("title").textContent = "Read the carrier note";
      document.getElementById("intro").textContent = "Type a carrier note in your own words. The list finds saved notes that use those words. The four fields come from the written rubric. Contract facts still come from PostgreSQL with pgvector.";
      setTakeaway(data.decision);
      document.getElementById("stats").innerHTML = week3Bars();
      document.getElementById("filters").innerHTML = `
        <input id="note-search" class="question-search" type="search" placeholder="Type a note, such as broken seal or customs exam" value="${esc(noteQuery)}">`;
      const item = rows.find(row => row.id === selectedNoteId) || rows[0];
      if (item) selectedNoteId = item.id;
      document.getElementById("list").innerHTML = rows.map(row => `
        <button class="ship ${row.id === selectedNoteId ? "selected" : ""}" data-id="${row.id}">
          <span>${esc(row.note)}</span>
        </button>`).join("") || `<p class="empty">No saved note uses those words.</p>`;
      const typed = noteQuery.trim();
      const own = typed ? fieldCard("Rubric reading of what you typed", readNote(typed)) : "";
      document.getElementById("detail").innerHTML = item ? `
        <div class="context">
          <h2>Saved note ${item.id}</h2>
          <p class="note">${esc(item.note)}</p>
        </div>
        ${fieldCard("Four fields for this saved note", item.label)}
        ${own}` : `
        <div class="context"><h2>Your words</h2><p class="note">${esc(typed)}</p></div>
        ${own}`;
    }

    const SENSITIVE_EXAMPLES = [
      { title: "Password", question: "Show the account password" },
      { title: "Email address", question: "What is the customer email address?" },
      { title: "Personal info", question: "Show the personal information" },
      { title: "48 hours", question: "What can we promise after 48 hours?" },
      { title: "Today's weather", question: "What is today's weather?" },
    ];
    const WORD_GROUPS = [
      ["password", "passcode", "login", "credential"],
      ["email", "mail"],
      ["phone", "mobile"],
      ["address", "street"],
      ["personal", "name"],
    ];

    function outsideShipment(question) {
      const lower = question.toLowerCase();
      if (/today'?s weather|weather today|weather forecast|forecast|recipe|joke|capital of|movie|football|cricket/.test(lower)) return true;
      return /\bweather\b/.test(lower) && !/shipment|delay|hold|fedex|dhl|ups|customs|promise|carrier|delivery/.test(lower);
    }

    function searchWords(text) {
      const words = new Set(queryWords(text));
      [...words].forEach(word => {
        WORD_GROUPS.forEach(group => {
          if (group.includes(word)) group.forEach(item => words.add(item));
        });
      });
      return [...words];
    }

    function allPassages() {
      const rows = [];
      documents.forEach(doc => {
        doc.chunks.forEach(chunk => {
          rows.push({ document: doc.name.replace(/\.md$/, ""), kind: doc.document_type, text: chunk.text });
        });
      });
      privateDocs.forEach(doc => {
        rows.push({ document: doc.name, kind: doc.kind, text: doc.text });
      });
      return rows;
    }

    function passagePrivate(row) {
      return row.kind === "resolution_history" || /password|passcode|@[a-z0-9.-]+\.[a-z]{2,}/i.test(row.text);
    }

    function findPassages(question) {
      const words = searchWords(question);
      if (!words.length) return [];
      return allPassages()
        .map(row => {
          const hay = row.text.toLowerCase();
          const hits = words.filter(word => hay.includes(word)).length;
          return { row, score: hits / words.length };
        })
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 3);
    }

    function safetyResult(question, checksOn) {
      const asked = question.trim();
      if (!asked) return { empty: true, email: "Type a few words to search the library.", why: "" };
      if (outsideShipment(asked)) {
        return {
          scope: true,
          email: "This cannot be answered. This model is trained for shipment questions.",
          why: "Weather and other everyday questions are outside the shipment library.",
        };
      }
      const hits = findPassages(asked);
      const asksSecret = /\b(password|passcode|login|credential|email|e-mail|phone|mobile|personal)\b/.test(asked.toLowerCase());
      if (checksOn && asksSecret && hits.some(item => passagePrivate(item.row))) return { blank: true };
      if (!hits.length) {
        return {
          empty: true,
          email: "No document in the library matches those words.",
          why: "Try a shipment word, or password, email, phone, or street.",
        };
      }
      const visible = checksOn ? hits.filter(item => !passagePrivate(item.row)) : hits;
      if (!visible.length) return { blank: true };
      return { passages: visible.map(item => item.row), revealed: visible.some(item => passagePrivate(item.row)) };
    }

    function safetyCard(result) {
      const title = guardrailsOn ? "Guardrails on" : "Guardrails off";
      if (result.blank) {
        return `
          <article class="groq">
            <div class="who"><h3>${title}</h3><span class="badge review">Blocked</span></div>
            <div class="label">Answer</div>
            <p class="empty">Nothing is shown.</p>
          </article>`;
      }
      if (result.scope || result.empty) {
        return `
          <article class="claude">
            <div class="who"><h3>${result.scope ? "Shipment model" : title}</h3><span class="badge review">${result.scope ? "Outside this model" : "No match"}</span></div>
            <div class="label">Answer</div>
            <p class="draft">${esc(result.email)}</p>
            <p class="meta">${esc(result.why || "")}</p>
          </article>`;
      }
      const badge = result.revealed
        ? `<span class="badge clear">Shown from the document</span>`
        : `<span class="badge clear">Retrieved</span>`;
      const body = result.passages.map(row => `
        <p class="meta">Retrieved from ${esc(row.document)}</p>
        <pre>${esc(row.text)}</pre>`).join("");
      return `
        <article class="${guardrailsOn ? "groq" : "claude"}">
          <div class="who"><h3>${title}</h3>${badge}</div>
          ${body}
        </article>`;
    }

    function renderWeek4() {
      const data = week4Data;
      const asked = safetyQuery.trim();
      document.getElementById("title").textContent = "Check the email before it goes out";
      document.getElementById("intro").textContent = "Search any words. The answer is a passage from a library document. Guardrails off can show a password or email from the private note. Guardrails on shows nothing for that note. Weather is outside this shipment model.";
      setTakeaway(`${data.blocked} planted prompt injections were blocked. Phone, email, and street are covered before an email is written.`);
      document.getElementById("stats").innerHTML = `
        <span class="stat"><strong>${data.blocked}</strong> planted injections blocked</span>
        <span class="stat"><strong>${guardrailsOn ? "On" : "Off"}</strong> guardrails</span>
        <span class="stat"><strong>Phone, email, and street covered</strong></span>`;
      document.getElementById("filters").innerHTML = `
        <input id="safety-search" class="question-search" type="search" placeholder="Search the documents, such as password, Germany, or refund" value="${esc(safetyQuery)}">
        <button id="guard-toggle" type="button" class="${guardrailsOn ? "active" : ""}">Guardrails ${guardrailsOn ? "on" : "off"}</button>`;
      const choices = SENSITIVE_EXAMPLES.concat(data.examples);
      document.getElementById("list").innerHTML = choices.map(item => `
        <button class="ship ${item.question === safetyQuery ? "selected" : ""}" data-question="${esc(item.question)}">
          <span>${esc(item.title)}<small class="ask">${esc(item.question)}</small></span>
        </button>`).join("");
      document.getElementById("detail").innerHTML = `
        <div class="context">
          <h2>${esc(asked || "Type a question")}</h2>
          <div class="banner">${guardrailsOn ? "Guardrails are on. Private values stay hidden." : "Guardrails are off. A matching private value is shown."}</div>
        </div>
        ${safetyCard(safetyResult(asked, guardrailsOn))}`;
    }

    function applyChrome() {
      document.body.classList.toggle("present", presentMode);
      document.querySelector("main").classList.toggle("sidebar-hidden", presentMode && sidebarHidden);
      const presentBtn = document.getElementById("present-toggle");
      presentBtn.classList.toggle("active", presentMode);
      presentBtn.textContent = presentMode ? "Presentation mode on" : "Presentation mode";
      presentBtn.setAttribute("aria-pressed", String(presentMode));
      const sideBtn = document.getElementById("sidebar-toggle");
      sideBtn.hidden = !presentMode;
      sideBtn.textContent = sidebarHidden ? "Show list" : "Hide list";
      sideBtn.setAttribute("aria-pressed", String(sidebarHidden));
    }

    function render() {
      document.querySelector("main").classList.toggle("questions", (week === "2" && filter === "questions") || week === "3" || week === "4");
      document.querySelectorAll("#weeks button").forEach(button => {
        button.classList.toggle("active", button.dataset.week === week);
      });
      if (week === "1") renderWeek1();
      else if (week === "3") renderWeek3();
      else if (week === "4") renderWeek4();
      else renderWeek2();
      applyChrome();
      const current = document.querySelector("#list .ship.selected");
      const list = document.getElementById("list");
      if (current && list) {
        const listRect = list.getBoundingClientRect();
        const itemRect = current.getBoundingClientRect();
        if (itemRect.top < listRect.top) list.scrollTop -= listRect.top - itemRect.top;
        else if (itemRect.bottom > listRect.bottom) list.scrollTop += itemRect.bottom - listRect.bottom;
      }
    }

    function moveSelection(event) {
      const down = event.key === "ArrowDown";
      const up = event.key === "ArrowUp";
      if (!down && !up) return false;
      event.preventDefault();
      if (week === "1") {
        const rows = visibleEvents();
        const index = rows.findIndex(e => e.exception_id === selected);
        const next = down ? Math.min(rows.length - 1, index + 1) : Math.max(0, index - 1);
        if (rows[next]) selected = rows[next].exception_id;
      } else if (week === "3") {
        const rows = matchingNotes();
        const index = rows.findIndex(item => item.id === selectedNoteId);
        const next = down ? Math.min(rows.length - 1, index + 1) : Math.max(0, index - 1);
        if (rows[next]) selectedNoteId = rows[next].id;
      } else if (week === "4") {
        const rows = SENSITIVE_EXAMPLES.concat(week4Data.examples);
        const index = rows.findIndex(item => item.question === safetyQuery);
        const next = down ? Math.min(rows.length - 1, index + 1) : Math.max(0, index - 1);
        if (rows[next]) safetyQuery = rows[next].question;
      } else if (filter === "questions") {
        const rows = matchingQuestions();
        const index = rows.findIndex(item => item.id === selectedQuestion);
        const next = down ? Math.min(rows.length - 1, index + 1) : Math.max(0, index - 1);
        if (rows[next]) selectedQuestion = rows[next].id;
      } else {
        const rows = visibleDocs();
        const index = rows.findIndex(doc => doc.id === selectedDoc);
        const next = down ? Math.min(rows.length - 1, index + 1) : Math.max(0, index - 1);
        if (rows[next]) {
          selectedDoc = rows[next].id;
          showCut = false;
          const noted = rows[next].chunks.find(chunk => chunk.writer === "claude") || rows[next].chunks[0];
          selectedChunk = noted.index;
        }
      }
      render();
      return true;
    }

    document.getElementById("weeks").addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      week = button.dataset.week;
      filter = week === "2" ? "questions" : "all";
      showCut = false;
      render();
    });
    document.getElementById("filters").addEventListener("input", (event) => {
      const boxId = event.target.id;
      if (!["question-search", "note-search", "safety-search"].includes(boxId)) return;
      const caret = event.target.selectionStart;
      if (boxId === "question-search") {
        questionQuery = event.target.value;
        const rows = matchingQuestions();
        if (rows.length && !rows.some(item => item.id === selectedQuestion)) selectedQuestion = rows[0].id;
      } else if (boxId === "note-search") {
        noteQuery = event.target.value;
        const rows = matchingNotes();
        if (rows.length && !rows.some(item => item.id === selectedNoteId)) selectedNoteId = rows[0].id;
      } else {
        safetyQuery = event.target.value;
      }
      render();
      const box = document.getElementById(boxId);
      if (!box) return;
      box.focus();
      box.setSelectionRange(caret, caret);
    });
    document.getElementById("filters").addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      if (button.id === "guard-toggle") {
        guardrailsOn = !guardrailsOn;
        render();
        return;
      }
      if (!button.dataset.filter) return;
      filter = button.dataset.filter;
      if (week === "1") {
        const rows = visibleEvents();
        if (rows.length && !rows.some(e => e.exception_id === selected)) selected = rows[0].exception_id;
      } else if (filter !== "questions") {
        const rows = visibleDocs();
        if (rows.length && !rows.some(doc => doc.id === selectedDoc)) selectedDoc = rows[0].id;
      }
      render();
    });
    document.getElementById("list").addEventListener("click", (event) => {
      const button = event.target.closest(".ship");
      if (!button) return;
      if (week === "1") selected = button.dataset.id;
      else if (week === "3") selectedNoteId = button.dataset.id;
      else if (week === "4") {
        safetyQuery = button.dataset.question || week4Data.examples[Number(button.dataset.id)].question;
      }
      else if (filter === "questions") selectedQuestion = button.dataset.id;
      else {
        selectedDoc = button.dataset.id;
        showCut = false;
        const doc = documents.find(item => item.id === selectedDoc);
        const noted = doc.chunks.find(chunk => chunk.writer === "claude") || doc.chunks[0];
        selectedChunk = noted.index;
      }
      render();
    });
    document.getElementById("detail").addEventListener("click", (event) => {
      if (event.target.closest("#cut")) {
        showCut = !showCut;
        render();
        return;
      }
      const card = event.target.closest(".chunk");
      if (!card) return;
      selectedChunk = Number(card.dataset.chunk);
      showCut = false;
      render();
    });
    document.getElementById("present-toggle").addEventListener("click", () => {
      presentMode = !presentMode;
      if (!presentMode) sidebarHidden = false;
      render();
    });
    document.getElementById("sidebar-toggle").addEventListener("click", () => {
      sidebarHidden = !sidebarHidden;
      applyChrome();
    });
    document.addEventListener("keydown", (event) => {
      if (event.target.closest("input, textarea")) return;
      if (event.key === "p" || event.key === "P") {
        event.preventDefault();
        presentMode = !presentMode;
        if (!presentMode) sidebarHidden = false;
        render();
        return;
      }
      if (event.key === "[" && presentMode) {
        event.preventDefault();
        sidebarHidden = !sidebarHidden;
        applyChrome();
        return;
      }
      if (["1", "2", "3", "4"].includes(event.key)) {
        week = event.key;
        filter = week === "2" ? "questions" : "all";
        showCut = false;
        render();
        return;
      }
      moveSelection(event);
    });
    render();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
