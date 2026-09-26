"""Day 2, step 1: write a short "where does this chunk come from" line for every chunk.

Run from the project root:  python src/contextualize.py
Output: data/contexts.json  ({chunk_id: context line})

Free-tier plan:
- Gemini Flash-Lite, the same model as the grader (free tier: about 500 requests/day, 15/minute).
- 5 neighbouring chunks per request, so 491 chunks take about 100 requests instead of 491.
  That leaves room for today's grading (about 130 requests).
- The model also sees the end of the chunk before and the start of the chunk after (the "nearby text").
- Progress is saved after every request. If the daily limit is reached, the script stops cleanly:
  run it again tomorrow and it continues where it stopped.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

CHUNKS_FILE = Path("data/chunks.json")
OUT_FILE = Path("data/contexts.json")
CONTEXT_MODEL = "gemini-3.5-flash-lite"   # use the same exact name as JUDGE_MODEL in evaluate.py
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
BATCH = 5            # chunks per request
EDGE_WORDS = 150     # nearby text shown before and after the batch
PAUSE = 6            # seconds between requests: stays under 15 requests/minute

PROMPT = """You help a search engine over football rulebooks.
Document: {title}

Below is a passage from this document, split into numbered chunks, with some text before and after for reference.
For EACH chunk, write 1-2 short sentences that situate it in the document, so the chunk can be found by search
even when read on its own. Mention:
- the document name,
- the Law / Article / section number and title the chunk belongs to (if you can tell),
- the main topic of the chunk.
If a chunk ends one article and starts another, name both.
Do not repeat the chunk's content in detail. Do not invent article numbers you cannot see.

Return JSON only: {"contexts": [{"id": "<chunk id>", "context": "<1-2 sentences>"}, ...]}

Text before (for reference only):
...{before}

{chunks}

Text after (for reference only):
{after}..."""


class DailyLimitReached(Exception):
    pass


def ask_gemini(client, prompt):
    for attempt in range(5):
        try:
            reply = client.chat.completions.create(
                model=CONTEXT_MODEL, temperature=0, messages=[{"role": "user", "content": prompt}])
            text = reply.choices[0].message.content
            return json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        except Exception as err:
            message = str(err)
            if "429" in message and ("PerDay" in message or "per day" in message.lower()):
                raise DailyLimitReached(message[:200])
            wait = 30 * (attempt + 1)
            print(f"  error ({message[:120]}); retrying in {wait}s...")
            time.sleep(wait)
    return None


def main():
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    contexts = json.loads(OUT_FILE.read_text(encoding="utf-8")) if OUT_FILE.exists() else {}
    client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_URL)

    # keep each document's chunks in reading order
    docs = {}
    for c in chunks:
        docs.setdefault(c["source"], []).append(c)

    todo = sum(1 for c in chunks if c["id"] not in contexts)
    print(f"{len(chunks) - todo}/{len(chunks)} chunks already done, {todo} to go "
          f"(~{-(-todo // BATCH)} requests).")

    try:
        for title, doc_chunks in docs.items():
            for start in range(0, len(doc_chunks), BATCH):
                batch = doc_chunks[start:start + BATCH]
                if all(c["id"] in contexts for c in batch):
                    continue
                before = " ".join(doc_chunks[start - 1]["text"].split()[-EDGE_WORDS:]) if start > 0 else "(start of document)"
                nxt = start + BATCH
                after = " ".join(doc_chunks[nxt]["text"].split()[:EDGE_WORDS]) if nxt < len(doc_chunks) else "(end of document)"
                numbered = "\n\n".join(f'[chunk id: {c["id"]}]\n{c["text"]}' for c in batch)
                prompt = (PROMPT.replace("{title}", title).replace("{before}", before)
                          .replace("{chunks}", numbered).replace("{after}", after))

                result = ask_gemini(client, prompt)
                wanted = {c["id"] for c in batch}
                for item in (result or {}).get("contexts", []):
                    if item.get("id") in wanted and item.get("context"):
                        contexts[item["id"]] = item["context"].strip()
                OUT_FILE.write_text(json.dumps(contexts, ensure_ascii=False, indent=2), encoding="utf-8")

                missed = [i for i in wanted if i not in contexts]
                print(f"{len(contexts)}/{len(chunks)} | {title[:40]}"
                      + (f" | {len(missed)} missed, will retry next run" if missed else ""))
                time.sleep(PAUSE)
    except DailyLimitReached as err:
        print(f"\nDaily Gemini limit reached ({err}).\nProgress is saved: run this script again tomorrow.")
        return

    left = sum(1 for c in chunks if c["id"] not in contexts)
    if left:
        print(f"\n{left} chunks still missing a context. Run the script again to fill them.")
    else:
        sample = chunks[len(chunks) // 2]
        print(f"\nAll done. Example ({sample['id']}):\n  {contexts[sample['id']]}")


if __name__ == "__main__":
    main()
