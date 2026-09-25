"""Step 3: baseline RAG - embed the chunks, find the closest ones, answer with an LLM.

Run from the project root:
    python src/rag.py "How many substitutes can a team use in a Champions League match?"

Note: embeddings are saved to data/embeddings_gemini.npy. If you re-run ingest.py, delete that file.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")  # avoid symbol errors in the Windows terminal

CHUNKS_FILE = Path("data/chunks.json")
EMB_FILE = Path("data/embeddings_gemini.npy")
EMBED_MODEL = "gemini-embedding-001"
EMBED_BATCH = 20                          # chunks sent per API call
EMBED_PAUSE = 13                          # seconds between batches: free tier = 100 texts/minute
LLM_MODEL = "openai/gpt-oss-120b"         # change here if Groq renames or retires it
TOP_K = 5                                 # number of chunks sent to the LLM

SYSTEM_PROMPT = """You answer questions about football rules and regulations.
Use ONLY the numbered context passages. After each fact, cite its passage like [2].
If the passages do not contain the answer, reply exactly:
"I don't know based on the provided documents."
Be concise."""


class BaselineRAG:
    def __init__(self):
        self.chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        self.gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.vectors = self._load_or_build_vectors()
        self.llm = Groq(api_key=os.environ["GROQ_API_KEY"])

    def _embed(self, texts, task_type):
        """Call Gemini, retrying with longer waits if we hit the rate limit."""
        for attempt in range(6):
            try:
                result = self.gemini.models.embed_content(
                    model=EMBED_MODEL,
                    contents=texts,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                vectors = np.array([e.values for e in result.embeddings], dtype=np.float32)
                return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # normalize to length 1
            except Exception as err:
                wait = 10 * (attempt + 1)
                print(f"  Gemini error ({err}); retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError("Gemini embedding failed after several retries.")

    def _load_or_build_vectors(self):
        if EMB_FILE.exists():
            vectors = np.load(EMB_FILE)
            if len(vectors) == len(self.chunks):
                return vectors
        print("Embedding chunks with Gemini (first run only)...")
        texts = [c["text"] for c in self.chunks]
        parts = []
        for start in range(0, len(texts), EMBED_BATCH):
            parts.append(self._embed(texts[start:start + EMBED_BATCH], "RETRIEVAL_DOCUMENT"))
            print(f"  {min(start + EMBED_BATCH, len(texts))}/{len(texts)}")
            time.sleep(EMBED_PAUSE)
        vectors = np.vstack(parts)
        np.save(EMB_FILE, vectors)
        return vectors

    def retrieve(self, question, k=TOP_K):
        q = self._embed([question], "RETRIEVAL_QUERY")[0]
        scores = self.vectors @ q          # cosine similarity (vectors are normalized)
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def generate(self, question, contexts):
        context_text = "\n\n".join(
            f"[{i}] ({c['source']}, p.{c['page_start']}-{c['page_end']})\n{c['text']}"
            for i, c in enumerate(contexts, start=1)
        )
        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
        )
        answer = response.choices[0].message.content.strip()
        return re.sub(r"【(\d+)[^】]*】", r"[\1]", answer)  # turn the model's 【1†L1-L4】 style into [1]

    def ask(self, question):
        contexts = self.retrieve(question)
        return {"question": question, "answer": self.generate(question, contexts), "contexts": contexts}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many substitutes can a team use in a UEFA Champions League match?"
    rag = BaselineRAG()
    result = rag.ask(question)

    print(f"\nQUESTION: {question}\n")
    print(f"ANSWER:\n{result['answer']}\n")
    print("RETRIEVED CHUNKS:")
    for i, c in enumerate(result["contexts"], start=1):
        print(f"[{i}] {c['score']:.3f} | {c['source']} | p.{c['page_start']}-{c['page_end']}")