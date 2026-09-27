"""Step 3: baseline RAG - embed the chunks, find the closest ones, answer with an LLM.

Run from the project root:
    python src/rag.py "How many substitutes can a team use in a Champions League match?"

Note: embeddings are saved to data/embeddings_gemini.npy (Day 1) and data/embeddings_contextual.npy (Day 2).
If you re-run ingest.py, delete those files.
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
    """Day 1: embed the raw chunk text."""
    emb_file = EMB_FILE

    def index_text(self, chunk):
        """The text that gets embedded for search."""
        return chunk["text"]

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
        if self.emb_file.exists():
            vectors = np.load(self.emb_file)
            if len(vectors) == len(self.chunks):
                return vectors
        print(f"Embedding chunks with Gemini into {self.emb_file} (first run only, ~7 min)...")
        texts = [self.index_text(c) for c in self.chunks]
        parts = []
        for start in range(0, len(texts), EMBED_BATCH):
            parts.append(self._embed(texts[start:start + EMBED_BATCH], "RETRIEVAL_DOCUMENT"))
            print(f"  {min(start + EMBED_BATCH, len(texts))}/{len(texts)}")
            time.sleep(EMBED_PAUSE)
        vectors = np.vstack(parts)
        np.save(self.emb_file, vectors)
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


class ContextualRAG(BaselineRAG):
    """Day 2: embed "context line + chunk text", so each chunk says where it comes from.
    Only SEARCH changes; the LLM still receives the same chunk text as Day 1 (one change at a time)."""
    emb_file = Path("data/embeddings_contextual.npy")
    contexts_file = Path("data/contexts.json")

    def __init__(self):
        if not self.contexts_file.exists():
            raise SystemExit("data/contexts.json not found. Run: python src/contextualize.py")
        self.contexts = json.loads(self.contexts_file.read_text(encoding="utf-8"))
        super().__init__()
        missing = [c["id"] for c in self.chunks if c["id"] not in self.contexts]
        if missing:
            raise SystemExit(f"{len(missing)} chunks have no context yet. Run contextualize.py again.")

    def index_text(self, chunk):
        return f"{self.contexts[chunk['id']]}\n\n{chunk['text']}"


# ---------------- Day 3: hybrid search + reranking ----------------
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # cross-encoder: reads (question, chunk) together
SHORTLIST = 20                             # merged candidates the reranker reads
POOL = 50                                  # how far down each list we look before merging
RRF_K = 60                                 # standard constant in Reciprocal Rank Fusion
STOPWORDS = set("""a an the of to in on at for by with from and or is are was were be been being it its this that
these those if then than as can may must shall will would should do does did has have had not no what which who
whom when where why how there their they them he she his her our we you your i""".split())


def tokenize(text):
    """Lowercase words for keyword search, without very common words."""
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


class HybridRerankRAG(ContextualRAG):
    """Day 3: vector search + BM25 keyword search, merged with RRF, then a reranker picks the final 5.
    Both searches and the reranker see "context line + chunk" (built on Day 2).
    The LLM still receives the same chunk text as Days 1-2."""

    def __init__(self):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder
        super().__init__()
        self.index_texts = [self.index_text(c) for c in self.chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.index_texts])
        print(f"Loading reranker {RERANK_MODEL} (first run downloads it)...")
        self.reranker = CrossEncoder(RERANK_MODEL, max_length=512)
        import torch
        if torch.cuda.is_available():
            self.reranker.model.half()   # half precision: ~1.1 GB instead of ~2.3 GB, fits a 4 GB GPU
            print(f"Reranker running on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Reranker running on CPU (slower, but works).")

    def retrieve(self, question, k=TOP_K):
        # 1) vector search (meaning)
        q = self._embed([question], "RETRIEVAL_QUERY")[0]
        vector_rank = np.argsort(self.vectors @ q)[::-1][:POOL]
        # 2) BM25 (exact words)
        bm25_rank = np.argsort(self.bm25.get_scores(tokenize(question)))[::-1][:POOL]
        # 3) merge the two lists with RRF: a chunk ranked high in either list ends up high
        fused = {}
        for ranking in (vector_rank, bm25_rank):
            for rank, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
        shortlist = sorted(fused, key=fused.get, reverse=True)[:SHORTLIST]
        # 4) rerank the shortlist and keep the best k
        scores = self.reranker.predict([(question, self.index_texts[i]) for i in shortlist])
        best = sorted(zip(shortlist, scores), key=lambda x: x[1], reverse=True)[:k]
        return [{**self.chunks[i], "score": float(s)} for i, s in best]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many substitutes can a team use in a UEFA Champions League match?"
    rag = HybridRerankRAG()
    result = rag.ask(question)

    print(f"\nQUESTION: {question}\n")
    print(f"ANSWER:\n{result['answer']}\n")
    print("RETRIEVED CHUNKS:")
    for i, c in enumerate(result["contexts"], start=1):
        print(f"[{i}] {c['score']:.3f} | {c['source']} | p.{c['page_start']}-{c['page_end']}")
