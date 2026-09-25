"""Step 4: build the test set (the "exam" every version of the RAG will take).

Run from the project root:  python src/make_testset.py
Output: eval/testset.json  ->  REVIEW IT BY HAND before using it.

Question types:
  single     - answer is in one chunk
  multi_hop  - answer needs two chunks from two different documents
  no_answer  - not in the documents; the right reply is "I don't know"
"""
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from rag import LLM_MODEL, BaselineRAG

OUT_FILE = Path("eval/testset.json")
random.seed(42)  # same questions every run

LOTG = "IFAB Laws of the Game 2026/27"
UCL = "UEFA Champions League Regulations 2026/27"
DISC = "UEFA Disciplinary Regulations"
RSTP = "FIFA Regulations on the Status and Transfer of Players"

SINGLE_TARGETS = {LOTG: 5, UCL: 5, DISC: 4, RSTP: 4}                        # 18 questions
MULTI_PAIRS = [(LOTG, UCL), (LOTG, UCL), (UCL, DISC), (UCL, DISC),
               (LOTG, DISC), (LOTG, DISC), (RSTP, UCL), (RSTP, UCL)]         # 8 questions
NO_ANSWER = [                                                                 # 4 questions
    "How many teams are relegated from the English Premier League each season?",
    "How long is each quarter in an NBA basketball game?",
    "Which club won the 2025/26 UEFA Champions League?",
    "How much does a ticket for the 2027 Champions League final cost?",
]
IDK = "I don't know based on the provided documents."

SINGLE_PROMPT = """You write test questions for a football-rules assistant.
Read the passage and write ONE question a football fan, coach or club official might ask,
whose answer is clearly and fully stated in the passage.
Rules:
- Ask about a concrete rule, number, condition or consequence (not a definition list or table of contents).
- Phrase it naturally, naming the competition or body only as a real person would (e.g. "in the Champions League").
- Never mention "the passage" or "the document".
- The answer must be short (1-2 sentences) and use only facts from the passage.
If the passage has no clear rule worth asking about, return {"skip": true}.
Return JSON only: {"question": "...", "answer": "..."}

Source: {source}
Passage:
{text}"""

MULTI_PROMPT = """You write hard test questions for a football-rules assistant.
Write ONE question that can only be answered by combining facts from BOTH passages below
(e.g. what happens on the pitch AND what the disciplinary consequence is).
Rules:
- Phrase it naturally, as a fan, coach or club official would ask it.
- Never mention "the passage" or "the document".
- The answer (1-3 sentences) must use facts from both passages and nothing else.
If the passages cannot be combined into a sensible question, return {"skip": true}.
Return JSON only: {"question": "...", "answer": "..."}

Passage A ({source_a}):
{text_a}

Passage B ({source_b}):
{text_b}"""


def ask_llm_json(rag, prompt):
    """Send a prompt to Groq and read the JSON it returns (retries on errors/limits)."""
    for attempt in range(5):
        try:
            response = rag.llm.chat.completions.create(
                model=LLM_MODEL, temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0)) if match else {"skip": True}
        except json.JSONDecodeError:
            return {"skip": True}
        except Exception as err:
            wait = 15 * (attempt + 1)
            print(f"  Groq error ({err}); retrying in {wait}s...")
            time.sleep(wait)
    return {"skip": True}


def fill(template, **values):
    # str.format would break on the JSON braces in the prompt, so replace fields by hand
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def main():
    rag = BaselineRAG()
    chunks, vectors = rag.chunks, rag.vectors
    by_source = {}
    for i, c in enumerate(chunks):
        by_source.setdefault(c["source"], []).append(i)

    testset, used = [], set()

    # 1) single-chunk questions
    for source, target in SINGLE_TARGETS.items():
        candidates = [i for i in by_source[source] if chunks[i]["page_start"] > 3]  # skip cover/contents
        random.shuffle(candidates)
        made = 0
        for i in candidates:
            if made == target:
                break
            out = ask_llm_json(rag, fill(SINGLE_PROMPT, source=source, text=chunks[i]["text"]))
            time.sleep(2)
            if out.get("skip") or not out.get("question"):
                continue
            made += 1
            used.add(i)
            testset.append({"type": "single", "question": out["question"],
                            "ground_truth": out["answer"], "source_chunk_ids": [chunks[i]["id"]]})
            print(f"single    | {source[:30]:30} | {out['question']}")

    # 2) multi-hop questions: a random chunk from doc A + the most similar chunk from doc B
    for source_a, source_b in MULTI_PAIRS:
        for _ in range(8):  # a few tries per pair
            a = random.choice([i for i in by_source[source_a] if chunks[i]["page_start"] > 3 and i not in used])
            b_ids = by_source[source_b]
            b = b_ids[int(np.argmax(vectors[b_ids] @ vectors[a]))]
            out = ask_llm_json(rag, fill(MULTI_PROMPT, source_a=source_a, text_a=chunks[a]["text"],
                                         source_b=source_b, text_b=chunks[b]["text"]))
            time.sleep(2)
            if out.get("skip") or not out.get("question"):
                continue
            used.update([a, b])
            testset.append({"type": "multi_hop", "question": out["question"], "ground_truth": out["answer"],
                            "source_chunk_ids": [chunks[a]["id"], chunks[b]["id"]]})
            print(f"multi_hop | {source_a[:14]} + {source_b[:14]} | {out['question']}")
            break

    # 3) questions the documents cannot answer
    for q in NO_ANSWER:
        testset.append({"type": "no_answer", "question": q, "ground_truth": IDK, "source_chunk_ids": []})

    for n, item in enumerate(testset, start=1):
        item["id"] = f"q{n:02d}"
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(testset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(testset)} questions to {OUT_FILE}. Review them by hand before evaluating.")


if __name__ == "__main__":
    main()
