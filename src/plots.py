"""Draw the progress charts as separate PNG images (for LinkedIn / README).

Run from the project root after each evaluation:  python src/plots.py
Output: results/figures/
  1_answer_correctness.png   answer correctness per day: simple, multi-hop, all
  2_all_metrics.png          the 4 metrics per day, one small chart each
  3_question_map.png         every question x every day: right / partial / wrong
"""
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # draw to files, no window
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

TESTSET = Path("eval/testset.json")
RESULTS = Path("results")
FIGURES = RESULTS / "figures"

DAYS = [   # same plan as dashboard.py
    ("day1_baseline", "Day 1", "Baseline"),
    ("day2_contextual", "Day 2", "Contextual\nretrieval"),
    ("day3_hybrid", "Day 3", "Hybrid +\nreranking"),
    ("day4_corrective", "Day 4", "Corrective\nRAG"),
    ("day5_agentic", "Day 5", "Agentic\nRAG"),
]

if (RESULTS / "day5_agentic_v2_scores.json").exists():   # show the second Day 5 run when it exists
    DAYS[-1] = ("day5_agentic_v2",) + DAYS[-1][1:]
METRICS = [
    ("answer_correctness", "Answer correctness"),
    ("context_recall", "Context recall"),
    ("retrieval_hit", "Retrieval hit"),
    ("faithfulness", "Faithfulness"),
]

# Colours (same family as the dashboard)
BG, INK, MUTED, GRID = "#F2F4EF", "#18261F", "#5E6D64", "#DDE3DA"
PITCH, YELLOW, RED, SOFT = "#1D5139", "#E3B21E", "#CF3A32", "#8FB8A0"

plt.rcParams.update({
    "font.family": ["Segoe UI", "Barlow", "DejaVu Sans"],
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
})


def load_runs():
    runs = []
    for key, label, technique in DAYS:
        path = RESULTS / f"{key}_scores.json"
        scores = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        runs.append({"key": key, "label": label, "technique": technique, "scores": scores})
    return runs


def average(scores, metric, qtype=None):
    values = [r.get(metric) for r in scores.values()
              if r["type"] != "no_answer" and (qtype is None or r["type"] == qtype) and r.get(metric) is not None]
    return sum(values) / len(values) if values else None


def x_axis(ax, runs):
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{r['label']}\n{r['technique']}" for r in runs], fontsize=10)
    ax.set_xlim(-0.4, len(runs) - 0.6)


def title(fig, main, sub):
    fig.text(0.06, 0.94, main, fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.875, sub, fontsize=12, color=MUTED, va="top")


def plot_correctness(runs):
    fig, ax = plt.subplots(figsize=(10, 6.25), dpi=160)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.78, bottom=0.17)
    series = [("All answerable", None, PITCH, 3.2), ("Simple", "single", SOFT, 2.2), ("Multi-hop", "multi_hop", YELLOW, 2.2)]
    labels = []   # end-of-line value labels, spread out later so they don't overlap
    for name, qtype, colour, width in series:
        xs, ys = [], []
        for i, r in enumerate(runs):
            if r["scores"]:
                v = average(r["scores"], "answer_correctness", qtype)
                if v is not None:
                    xs.append(i)
                    ys.append(v)
        ax.plot(xs, ys, color=colour, linewidth=width, marker="o", markersize=8, label=name, zorder=3)
        if xs:
            labels.append([ys[-1], xs[-1], colour if colour != SOFT else "#4E8A67"])
    labels.sort(key=lambda l: l[0])
    for a, b in zip(labels, labels[1:]):
        if b[0] - a[0] < 0.03:
            b.append(a[0] + 0.03)
    for lab in labels:
        y_text = lab[3] if len(lab) > 3 else lab[0]
        ax.annotate(f"{lab[0]:.2f}", (lab[1], lab[0]), xytext=(lab[1] + 0.12, y_text), textcoords="data",
                    va="center", fontsize=11, fontweight="bold", color=lab[2])
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("Answer correctness (0 to 1)")
    ax.grid(axis="y", color=GRID, linewidth=1)
    x_axis(ax, runs)
    ax.legend(loc="lower left", frameon=False, fontsize=11, ncol=3)
    title(fig, "Is the answer right?", "Football-rules RAG · same 30-question exam every day · one new technique per day")
    fig.savefig(FIGURES / "1_answer_correctness.png")
    plt.close(fig)


def plot_all_metrics(runs):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=160, sharex=True)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.8, bottom=0.07, hspace=0.35, wspace=0.18)
    for ax, (metric, name) in zip(axes.flat, METRICS):
        for qtype, colour, width, label in [("single", SOFT, 1.8, "Simple"), ("multi_hop", YELLOW, 1.8, "Multi-hop"),
                                            (None, PITCH, 2.8, "All")]:
            xs = [i for i, r in enumerate(runs) if r["scores"]]
            ys = [average(runs[i]["scores"], metric, qtype) for i in xs]
            ax.plot(xs, ys, color=colour, linewidth=width, marker="o", markersize=5, label=label, zorder=3)
        last = [i for i, r in enumerate(runs) if r["scores"]][-1]
        ax.set_title(f"{name}   {average(runs[last]['scores'], metric):.2f}", loc="left", fontsize=13, fontweight="bold")
        ax.set_ylim(0.5, 1.05)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([r["label"] for r in runs], fontsize=9)
        ax.set_xlim(-0.3, len(runs) - 0.7)
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=9, ncol=3)
    title(fig, "Every metric, every day", "Finding: Context recall, Retrieval hit · Writing: Answer correctness, Faithfulness")
    fig.savefig(FIGURES / "2_all_metrics.png")
    plt.close(fig)


def plot_question_map(runs, testset):
    done = [r for r in runs if r["scores"]]
    questions = [q for q in testset if q["type"] != "no_answer"]
    questions.sort(key=lambda q: (q["type"] != "multi_hop", q["id"]))   # multi-hop on top
    grid = [[r["scores"].get(q["id"], {}).get("answer_correctness", float("nan")) for r in done] for q in questions]

    fig, ax = plt.subplots(figsize=(7.5, 10), dpi=160)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.03)
    cmap = ListedColormap([RED, YELLOW, PITCH])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(done)))
    ax.set_xticklabels([f"{r['label']}\n{r['technique']}" for r in done], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels([f"{q['id']}  {'multi-hop' if q['type'] == 'multi_hop' else 'simple'}" for q in questions], fontsize=9)
    ax.set_xticks([x - 0.5 for x in range(1, len(done))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(questions))], minor=True)
    ax.grid(which="minor", color=BG, linewidth=2)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    n_multi = sum(1 for q in questions if q["type"] == "multi_hop")
    ax.axhline(n_multi - 0.5, color=INK, linewidth=1.5)
    fig.text(0.06, 0.965, "Every question, every day", fontsize=18, fontweight="bold", va="top")
    fig.text(0.06, 0.935, "Green = right · yellow = partly right · red = wrong", fontsize=11, color=MUTED, va="top")
    fig.savefig(FIGURES / "3_question_map.png")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not any(r["scores"] for r in runs):
        raise SystemExit("No results yet. Run evaluate.py first.")
    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    plot_correctness(runs)
    plot_all_metrics(runs)
    plot_question_map(runs, testset)
    print(f"Charts saved in {FIGURES}/")


if __name__ == "__main__":
    main()
