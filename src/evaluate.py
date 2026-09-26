"""Step 5: give the exam - the RAG answers the test set, then Ragas grades it.

Run from the project root:  python src/evaluate.py --run day1_baseline
Outputs (in results/):
  <run>_answers.json   the RAG's answers + retrieved chunks
  <run>_scores.json    the score of every question
  <run>_summary.csv    the score table (your screenshot)

Progress is saved after every question, so if a rate limit stops the script,
just run the same command again and it continues where it stopped.
"""
import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import sys
import types

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

# Workaround for a known Ragas 0.4.3 bug: it imports a module that newer langchain-community removed.
# We register an empty stand-in so the import succeeds (we don't use Vertex AI anyway).
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _stub

from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall, Faithfulness

from rag import BaselineRAG, ContextualRAG

load_dotenv()

TESTSET = Path("eval/testset.json")
RESULTS = Path("results")
JUDGE_MODEL = "gemini-3.5-flash-lite"  # the "grader" LLM: a different family from the RAG's LLM, so it doesn't grade itself
RAG_VERSIONS = {                                  # we add one line here each day
    "day1_baseline": BaselineRAG,
    "day2_contextual": ContextualRAG,
}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

CORRECTNESS_PROMPT = """You grade an answer to a football-rules question against an answer key.
- "correct": the answer contains the key facts of the answer key and contradicts none of them.
  Extra details, different wording and citations are fine.
- "partial": some key facts are missing, but nothing contradicts the answer key.
- "incorrect": the answer contradicts the key, misses its main fact, or says it doesn't know.
Return JSON only: {"verdict": "correct" | "partial" | "incorrect", "reason": "<one short sentence>"}

Question: {question}
Answer key: {reference}
Answer to grade: {response}"""
VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def judge_correctness(client, question, reference, response):
    """Our own simple grader: one LLM call, one verdict (more reliable than Ragas' claim-splitting here)."""
    prompt = (CORRECTNESS_PROMPT.replace("{question}", question)
              .replace("{reference}", reference).replace("{response}", response))
    reply = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0, messages=[{"role": "user", "content": prompt}])
    text = reply.choices[0].message.content
    verdict = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
    return VERDICT_SCORE[verdict["verdict"].strip().lower()], verdict.get("reason", "")


def load(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_citations(text):
    return re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", text).strip()


def with_retries(fn, what):
    """Run fn(); if an API limit or error hits, wait and try again."""
    for attempt in range(6):
        try:
            return fn()
        except Exception as err:
            wait = 20 * (attempt + 1)
            print(f"  {what} failed ({str(err)[:120]}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after several retries.")


# ---------- 1) the RAG answers every question ----------
def answer_all(run, testset):
    path = RESULTS / f"{run}_answers.json"
    answers = load(path, {})
    rag = None
    for q in testset:
        if q["id"] in answers:
            continue
        rag = rag or RAG_VERSIONS[run]()
        result = with_retries(lambda: rag.ask(q["question"]), f"answering {q['id']}")
        answers[q["id"]] = {
            "answer": result["answer"],
            "contexts": [c["text"] for c in result["contexts"]],
            "context_ids": [c["id"] for c in result["contexts"]],
        }
        save(path, answers)
        print(f"answered {q['id']}")
        time.sleep(2)
    return answers


# ---------- 2) Ragas grades every answer ----------
async def score_one(q, a, metrics, grader):
    response = strip_citations(a["answer"])
    found = [cid in a["context_ids"] for cid in q["source_chunk_ids"]]
    row = {"id": q["id"], "type": q["type"]}

    if q["type"] == "no_answer":
        # the right behaviour is to refuse
        row["correct_refusal"] = float("don't know" in response.lower())
        return row

    row["retrieval_hit"] = sum(found) / len(found)   # share of the needed chunks that search found
    row["context_recall"] = (await metrics["recall"].ascore(
        user_input=q["question"], retrieved_contexts=a["contexts"], reference=q["ground_truth"])).value
    row["faithfulness"] = (await metrics["faith"].ascore(
        user_input=q["question"], response=response, retrieved_contexts=a["contexts"])).value
    row["answer_correctness"], row["correctness_reason"] = judge_correctness(
        grader, q["question"], q["ground_truth"], response)
    return row


def score_all(run, testset, answers, redo_correctness=False):
    path = RESULTS / f"{run}_scores.json"
    scores = load(path, {})
    key = os.environ["GEMINI_API_KEY"]
    judge = llm_factory(JUDGE_MODEL, provider="openai", client=AsyncOpenAI(api_key=key, base_url=GEMINI_URL))
    grader = OpenAI(api_key=key, base_url=GEMINI_URL)
    metrics = {"recall": ContextRecall(llm=judge), "faith": Faithfulness(llm=judge)}

    if redo_correctness:   # re-grade only the correctness column, keep the other scores
        for q in testset:
            row = scores.get(q["id"])
            if not row or q["type"] == "no_answer":
                continue
            response = strip_citations(answers[q["id"]]["answer"])
            row["answer_correctness"], row["correctness_reason"] = with_retries(
                lambda: judge_correctness(grader, q["question"], q["ground_truth"], response),
                f"grading {q['id']}")
            save(path, scores)
            print(f"regraded {q['id']}: {row['answer_correctness']:.1f} | {row['correctness_reason']}")
            time.sleep(5)

    for q in testset:
        if q["id"] in scores:
            continue
        row = with_retries(lambda: asyncio.run(score_one(q, answers[q["id"]], metrics, grader)),
                           f"scoring {q['id']}")
        scores[q["id"]] = row
        save(path, scores)
        print(f"scored   {q['id']}: " + ", ".join(f"{k}={v:.2f}" for k, v in row.items() if isinstance(v, float)))
        time.sleep(15)   # stay under the free-tier requests-per-minute limit
    return scores


# ---------- 3) the score table ----------
def summarise(run, scores):
    df = pd.DataFrame(scores.values())
    cols = ["retrieval_hit", "context_recall", "faithfulness", "answer_correctness"]
    answerable = df[df["type"] != "no_answer"]

    table = answerable.groupby("type")[cols].mean()
    table.loc["ALL answerable"] = answerable[cols].mean()
    refusal = df.loc[df["type"] == "no_answer", "correct_refusal"].mean()

    table = table.round(2)
    table.to_csv(RESULTS / f"{run}_summary.csv")
    print(f"\n===== {run} =====")
    print(table.to_string())
    print(f"\nCorrect 'I don't know' on unanswerable questions: {refusal:.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="day1_baseline", choices=RAG_VERSIONS)
    parser.add_argument("--redo-correctness", action="store_true", help="re-grade only answer_correctness")
    args = parser.parse_args()
    run = args.run

    RESULTS.mkdir(exist_ok=True)
    testset = load(TESTSET, [])
    answers = answer_all(run, testset)
    scores = score_all(run, testset, answers, args.redo_correctness)
    summarise(run, scores)


if __name__ == "__main__":
    main()
