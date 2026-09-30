"""Build the progress dashboard: results/dashboard.html (open it in your browser).

Run from the project root after each evaluation:  python src/dashboard.py
It reads eval/testset.json and results/<run>_scores.json / <run>_answers.json.
"""
import html
import json
from pathlib import Path

TESTSET = Path("eval/testset.json")
RESULTS = Path("results")
OUT = RESULTS / "dashboard.html"

# The 5-day plan: (run name used in evaluate.py, day label, technique)
DAYS = [
    ("day1_baseline", "Day 1", "Baseline"),
    ("day2_contextual", "Day 2", "Contextual retrieval"),
    ("day3_hybrid", "Day 3", "Hybrid + reranking"),
    ("day4_corrective", "Day 4", "Corrective RAG"),
    ("day5_agentic", "Day 5", "Agentic RAG"),
]

# Metric name, what it checks (plain words), which job it grades
METRICS = [
    ("answer_correctness", "Answer correctness", "Is the answer right?", "Writing"),
    ("context_recall", "Context recall", "Do the found chunks contain the full answer?", "Finding"),
    ("retrieval_hit", "Retrieval hit", "Did search find the exact chunks the answer came from?", "Finding"),
    ("faithfulness", "Faithfulness", "Does the answer only use facts from the chunks?", "Writing"),
]
TYPE_LABEL = {"single": "Simple", "multi_hop": "Multi-hop", "no_answer": "Unanswerable"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarise(scores):
    rows = list(scores.values())
    answerable = [r for r in rows if r["type"] != "no_answer"]
    out = {}
    for group, members in [("all", answerable),
                           ("single", [r for r in answerable if r["type"] == "single"]),
                           ("multi_hop", [r for r in answerable if r["type"] == "multi_hop"])]:
        out[group] = {m[0]: mean([r.get(m[0]) for r in members]) for m in METRICS}
    refusals = [r.get("correct_refusal") for r in rows if r["type"] == "no_answer"]
    out["refusal"] = (sum(1 for v in refusals if v == 1.0), len(refusals))
    return out


def grade(v):
    """Referee-card colour for a score."""
    if v is None:
        return "none"
    return "good" if v >= 0.9 else "warn" if v >= 0.75 else "bad"


def fmt(v):
    return "–" if v is None else f"{v:.2f}"


def build():
    testset = {q["id"]: q for q in load(TESTSET)}
    days = []
    for key, label, technique in DAYS:
        scores = load(RESULTS / f"{key}_scores.json")
        answers = load(RESULTS / f"{key}_answers.json") or {}
        days.append({"key": key, "label": label, "technique": technique, "scores": scores,
                     "answers": answers, "summary": summarise(scores) if scores else None})
    done = [d for d in days if d["summary"]]
    if not done:
        raise SystemExit("No results yet. Run evaluate.py first.")
    latest = done[-1]

    # ---------- scoreboard ----------
    head = "".join(
        f'<th class="{"" if d["summary"] else "future"}"><span class="day">{d["label"]}</span>'
        f'<span class="tech">{html.escape(d["technique"])}</span></th>' for d in days)
    body = ""
    for key, name, question, job in METRICS:
        cells, prev = "", None
        for d in days:
            v = d["summary"]["all"][key] if d["summary"] else None
            delta = ""
            if v is not None and prev is not None:
                diff = v - prev
                if abs(diff) >= 0.005:
                    delta = f'<span class="delta {"up" if diff > 0 else "down"}">{diff:+.2f}</span>'
            cells += (f'<td class="{grade(v)}"><span class="num">{fmt(v)}</span>{delta}</td>')
            prev = v if v is not None else prev
        body += (f'<tr><th scope="row"><span class="mname">{name}</span>'
                 f'<span class="mq">{question}</span><span class="job">{job}</span></th>{cells}</tr>')
    cells = ""
    for d in days:
        if d["summary"]:
            ok, n = d["summary"]["refusal"]
            cells += f'<td class="{grade(ok / n if n else None)}"><span class="num">{ok}/{n}</span></td>'
        else:
            cells += '<td class="none"><span class="num">–</span></td>'
    body += ('<tr><th scope="row"><span class="mname">Correct refusal</span>'
             '<span class="mq">On unanswerable questions, did it say "I don\'t know"?</span>'
             f'<span class="job">Honesty</span></th>{cells}</tr>')

    # ---------- simple vs multi-hop for the latest day ----------
    split = ""
    for key, name, _, _ in METRICS:
        bars = ""
        for group in ("single", "multi_hop"):
            v = latest["summary"][group][key]
            width = 0 if v is None else v * 100
            bars += (f'<div class="bar"><span class="blabel">{TYPE_LABEL[group]}</span>'
                     f'<span class="track"><span class="fill {grade(v)}" style="width:{width:.0f}%"></span></span>'
                     f'<span class="bval">{fmt(v)}</span></div>')
        split += f'<div class="metric"><h3>{name}</h3>{bars}</div>'

    # ---------- question explorer data ----------
    explorer = {}
    for d in done:
        rows = []
        for qid, q in testset.items():
            s = d["scores"].get(qid, {})
            a = d["answers"].get(qid, {})
            rows.append({"id": qid, "type": q["type"], "question": q["question"],
                         "key": q["ground_truth"], "answer": a.get("answer", ""),
                         "reason": s.get("correctness_reason", ""),
                         "scores": {m[0]: s.get(m[0]) for m in METRICS},
                         "refusal": s.get("correct_refusal"),
                         "trace": a.get("trace", [])})
        explorer[d["key"]] = rows
    options = "".join(f'<option value="{d["key"]}"{" selected" if d is latest else ""}>'
                      f'{d["label"]}: {html.escape(d["technique"])}</option>' for d in done)

    page = TEMPLATE
    for token, value in {"%HEAD%": head, "%BODY%": body, "%SPLIT%": split, "%OPTIONS%": options,
                         "%LATEST%": f'{latest["label"]}: {html.escape(latest["technique"])}',
                         "%DATA%": json.dumps(explorer, ensure_ascii=False).replace("</", "<\\/"),
                         "%METRICS%": json.dumps([[m[0], m[1]] for m in METRICS])}.items():
        page = page.replace(token, value)
    OUT.write_text(page, encoding="utf-8")
    print(f"Dashboard written to {OUT}. Open it in your browser.")


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Football Rules RAG: 5-day progress</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --pitch:#16402E; --pitch-2:#1D5139; --line:#E9F0EA; --chalk:#F2F4EF; --ink:#18261F; --muted:#5E6D64;
  --good:#2E8B57; --warn:#E3B21E; --bad:#CF3A32; --rule:#D8DED6;
  --display:"Barlow Condensed","Arial Narrow",sans-serif; --body:"Barlow","Segoe UI",Arial,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--chalk);color:var(--ink);font:16px/1.55 var(--body)}
.wrap{max-width:1180px;margin:0 auto;padding:0 28px}
header{background:var(--pitch);color:var(--line);padding:40px 0 34px;border-bottom:6px solid var(--line)}
header h1{font:700 52px/1 var(--display);margin:0 0 10px}
header p{margin:0;max-width:62ch;color:#BFD3C5;font-size:18px}
section{padding:40px 0 8px}
h2{font:600 30px/1.1 var(--display);margin:0 0 6px}
.lead{margin:0 0 20px;color:var(--muted);max-width:70ch}

.board{background:var(--pitch);border-radius:14px;padding:10px 18px 14px;overflow-x:auto;
  box-shadow:inset 0 0 0 2px var(--pitch-2)}
.board table{border-collapse:collapse;width:100%;min-width:860px;color:var(--line);table-layout:fixed}
.board thead th{text-align:center;padding:14px 8px 12px;border-bottom:2px solid rgba(233,240,234,.25);vertical-align:bottom}
.board thead th:first-child{text-align:left}
.day{display:block;font:700 22px/1 var(--display)}
.tech{display:block;font-size:13px;color:#A9C3B2;margin-top:4px}
.board thead th.future .day,.board thead th.future .tech{opacity:.45}
.board thead th:first-child{width:32%}
.board tbody th{text-align:left;padding:14px 8px;border-bottom:1px solid rgba(233,240,234,.12);font-weight:400}
.mname{display:block;font:600 21px/1.1 var(--display)}
.mq{display:block;font-size:14px;color:#BFD3C5;margin-top:3px}
.job{display:inline-block;margin-top:6px;font-size:12px;color:#16402E;background:#BFD3C5;border-radius:4px;padding:1px 7px}
.board td{text-align:center;padding:10px 6px;border-bottom:1px solid rgba(233,240,234,.12);position:relative}
.num{font:700 40px/1 var(--display);letter-spacing:.5px}
.board td.good .num{color:#9FE3B5}
.board td.warn .num{color:var(--warn)}
.board td.bad .num{color:#FF8B84}
.board td.none .num{color:rgba(233,240,234,.3)}
.board td.warn .num::after,.board td.bad .num::after{content:"";display:inline-block;width:10px;height:15px;border-radius:2px;margin-left:8px;vertical-align:6px}
.board td.warn .num::after{background:var(--warn)}
.board td.bad .num::after{background:var(--bad)}
.delta{display:block;font:600 14px/1.2 var(--body);margin-top:4px}
.delta.up{color:#9FE3B5}.delta.down{color:#FF8B84}
.legend{display:flex;gap:22px;flex-wrap:wrap;margin:12px 2px 0;font-size:14px;color:var(--muted)}
.legend i{display:inline-block;width:9px;height:13px;border-radius:2px;margin-right:7px;vertical-align:-1px}

.split{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}
.metric{background:#fff;border:1px solid var(--rule);border-radius:10px;padding:16px 18px}
.metric h3{font:600 20px/1.1 var(--display);margin:0 0 12px}
.bar{display:grid;grid-template-columns:78px 1fr 40px;align-items:center;gap:10px;margin:8px 0}
.blabel{font-size:14px;color:var(--muted)}
.track{height:12px;background:#E6EBE4;border-radius:6px;overflow:hidden}
.fill{display:block;height:100%;border-radius:6px}
.fill.good{background:var(--good)}.fill.warn{background:var(--warn)}.fill.bad{background:var(--bad)}
.bval{font:600 18px/1 var(--display);text-align:right}

.controls{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
select,button{font:500 15px var(--body);padding:8px 12px;border:1px solid var(--rule);border-radius:8px;background:#fff;color:var(--ink);cursor:pointer}
button[aria-pressed="true"]{background:var(--pitch);color:var(--line);border-color:var(--pitch)}
select:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid var(--warn);outline-offset:2px}
.q{background:#fff;border:1px solid var(--rule);border-radius:10px;margin-bottom:10px}
.q summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:48px 1fr auto;gap:14px;align-items:center;padding:14px 16px}
.q summary::-webkit-details-marker{display:none}
.qid{font:700 20px/1 var(--display);color:var(--muted)}
.qtext{font-size:15px}
.qtype{display:block;font-size:12px;color:var(--muted);margin-top:2px}
.chips{display:flex;gap:6px}
.chip{min-width:46px;text-align:center;font:600 16px/1 var(--display);padding:6px 6px;border-radius:6px;color:#fff}
.chip.good{background:var(--good)}.chip.warn{background:var(--warn);color:#3B2E05}.chip.bad{background:var(--bad)}.chip.none{background:#C9D0C7;color:#4A564F}
.detail{padding:0 16px 16px 78px;display:grid;gap:10px}
.detail h4{margin:0;font:600 15px var(--body);color:var(--muted)}
.detail p{margin:2px 0 0;max-width:80ch}
.chiplegend{font-size:13px;color:var(--muted);margin:0 0 12px}
footer{padding:36px 0 50px;color:var(--muted);font-size:14px}
@media (max-width:700px){header h1{font-size:38px}.q summary{grid-template-columns:40px 1fr}.chips{grid-column:1/-1}.detail{padding-left:16px}}
</style>
</head>
<body>
<header><div class="wrap">
  <h1>Football Rules RAG</h1>
  <p>An assistant that answers questions about the Laws of the Game, UEFA and FIFA rules. One new RAG technique each day, graded on the same 30-question exam.</p>
</div></header>

<main class="wrap">
<section>
  <h2>Scoreboard</h2>
  <p class="lead">Average over the 26 answerable questions. Scores go from 0 to 1. The small number under a score is the change from the previous day.</p>
  <div class="board"><table>
    <thead><tr><th><span class="tech">Metric</span></th>%HEAD%</tr></thead>
    <tbody>%BODY%</tbody>
  </table></div>
  <div class="legend">
    <span><i style="background:#9FE3B5"></i>0.90 or more</span>
    <span><i style="background:var(--warn)"></i>0.75 to 0.89: yellow card</span>
    <span><i style="background:var(--bad)"></i>below 0.75: red card</span>
  </div>
</section>

<section>
  <h2>Simple vs multi-hop questions</h2>
  <p class="lead">%LATEST%. Simple questions need one document; multi-hop questions need two.</p>
  <div class="split">%SPLIT%</div>
</section>

<section>
  <h2>Every question</h2>
  <p class="lead">Open a question to see the answer key, the assistant's answer and why the grader scored it that way.</p>
  <div class="controls">
    <label for="run" class="sr">Day</label>
    <select id="run">%OPTIONS%</select>
    <button data-f="all" aria-pressed="true">All</button>
    <button data-f="single" aria-pressed="false">Simple</button>
    <button data-f="multi_hop" aria-pressed="false">Multi-hop</button>
    <button data-f="no_answer" aria-pressed="false">Unanswerable</button>
    <button data-f="wrong" aria-pressed="false">Wrong answers</button>
  </div>
  <p class="chiplegend">Chips, left to right: answer correctness, context recall, retrieval hit, faithfulness.</p>
  <div id="list"></div>
</section>
</main>
<footer><div class="wrap">Documents: IFAB Laws of the Game 2026/27, UEFA Champions League Regulations 2026/27, UEFA Disciplinary Regulations, FIFA RSTP. Graded with Ragas and Gemini.</div></footer>

<script>
const DATA = %DATA%;
const METRICS = %METRICS%;
const TYPES = {single:"Simple", multi_hop:"Multi-hop", no_answer:"Unanswerable"};
let filter = "all";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const grade = v => v == null ? "none" : v >= 0.9 ? "good" : v >= 0.75 ? "warn" : "bad";
function render(){
  const rows = DATA[document.getElementById("run").value].filter(r =>
    filter === "all" ? true :
    filter === "wrong" ? (r.type === "no_answer" ? r.refusal === 0 : r.scores.answer_correctness < 1) :
    r.type === filter);
  document.getElementById("list").innerHTML = rows.length ? rows.map(r => {
    const chips = r.type === "no_answer"
      ? `<span class="chip ${grade(r.refusal)}">${r.refusal === 1 ? "Refused" : r.refusal === 0 ? "Guessed" : "–"}</span>`
      : METRICS.map(([k, name]) => `<span class="chip ${grade(r.scores[k])}" title="${name}">${r.scores[k] == null ? "–" : r.scores[k].toFixed(2)}</span>`).join("");
    return `<details class="q"><summary><span class="qid">${r.id}</span>
      <span class="qtext">${esc(r.question)}<span class="qtype">${TYPES[r.type]}</span></span>
      <span class="chips">${chips}</span></summary>
      <div class="detail"><div><h4>Answer key</h4><p>${esc(r.key)}</p></div>
      <div><h4>Assistant's answer</h4><p>${esc(r.answer) || "No answer saved."}</p></div>
      ${r.trace && r.trace.length ? `<div><h4>Checker (Corrective RAG)</h4>${r.trace.map(t =>
        `<p>Round ${t.round}: ${t.note ? esc(t.note) : t.complete ? "chunks judged complete" :
          `missing <em>${esc(t.missing)}</em> → searched <em>"${esc(t.follow_up_query)}"</em>`}</p>`).join("")}</div>` : ""}
      ${r.reason ? `<div><h4>Grader's reason</h4><p>${esc(r.reason)}</p></div>` : ""}</div></details>`;
  }).join("") : "<p>No questions match this filter.</p>";
}
document.getElementById("run").addEventListener("change", render);
document.querySelectorAll("[data-f]").forEach(b => b.addEventListener("click", () => {
  filter = b.dataset.f;
  document.querySelectorAll("[data-f]").forEach(x => x.setAttribute("aria-pressed", x === b));
  render();
}));
render();
</script>
<style>.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}</style>
</body>
</html>"""

if __name__ == "__main__":
    build()
