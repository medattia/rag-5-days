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


# ---------------- Day 4: Corrective RAG ----------------
CHECK_MODEL = "openai/gpt-oss-20b"   # the checker (Groq, its own free quota, separate from the answering model)
MAX_FOLLOWUPS = 2                   # at most 2 extra searches per question
MIN_NEW_SLOTS = 2                   # when searching again, keep at least 2 of the 5 places for new chunks
CHECK_PAUSE = 25                    # seconds after each check: free tier allows 8,000 tokens/minute

CHECK_PROMPT = """You check whether retrieved passages are enough to answer a football-rules question.

Question: {question}

Passages:
{passages}

Tasks:
1. List the numbers of the passages that are relevant to the question.
2. Decide if the relevant passages contain ALL the facts needed to answer EVERY part of the question.
   Multi-part questions (e.g. "what happens on the pitch AND what ban follows") need facts for each part.
3. If something is missing, describe it in a few words and write ONE short search query for the missing part only.
   Mention the rulebook if you can guess it (Laws of the Game, UEFA Champions League Regulations,
   UEFA Disciplinary Regulations, FIFA Regulations on the Status and Transfer of Players).

Return JSON only:
{"relevant": [1, 2], "complete": true, "missing": "", "follow_up_query": ""}"""


class CorrectiveRAG(HybridRerankRAG):
    """Day 4: after the Day 3 search, a checker (gpt-oss-20b) reviews the 5 chunks.
    If facts are missing, it writes a follow-up query; irrelevant chunks are dropped and the free places
    are filled with the follow-up's best results. Up to 2 follow-ups. The LLM still gets 5 chunks."""

    def check(self, question, chunks):
        passages = "\n\n".join(f"[{i}] ({c['source']})\n{c['text']}" for i, c in enumerate(chunks, start=1))
        prompt = CHECK_PROMPT.replace("{question}", question).replace("{passages}", passages)
        for attempt in range(5):
            try:
                reply = self.llm.chat.completions.create(
                    model=CHECK_MODEL, temperature=0, messages=[{"role": "user", "content": prompt}])
                time.sleep(CHECK_PAUSE)
                return json.loads(re.search(r"\{.*\}", reply.choices[0].message.content, re.DOTALL).group(0))
            except (json.JSONDecodeError, AttributeError):
                return None   # unreadable answer: keep the chunks as they are
            except Exception as err:
                wait = 30 * (attempt + 1)
                print(f"  checker error ({str(err)[:100]}); retrying in {wait}s...")
                time.sleep(wait)
        return None

    def retrieve(self, question, k=TOP_K):
        chunks = super().retrieve(question, k)
        trace = []
        for round_no in range(1, MAX_FOLLOWUPS + 1):
            verdict = self.check(question, chunks)
            if not verdict:
                trace.append({"round": round_no, "note": "check failed, chunks kept"})
                break
            query = (verdict.get("follow_up_query") or "").strip()
            step = {"round": round_no, "complete": bool(verdict.get("complete")),
                    "missing": verdict.get("missing", ""), "follow_up_query": query}
            trace.append(step)
            if verdict.get("complete") or not query:
                break

            # drop irrelevant chunks, keep at most k - MIN_NEW_SLOTS relevant ones (best-ranked first)
            numbers = [n for n in verdict.get("relevant", []) if isinstance(n, int) and 1 <= n <= len(chunks)]
            keep = [chunks[n - 1] for n in sorted(set(numbers))][:k - MIN_NEW_SLOTS]
            current_ids = {c["id"] for c in chunks}
            new = [c for c in super().retrieve(query, k) if c["id"] not in current_ids]
            added = new[:k - len(keep)]
            updated = keep + added
            for c in chunks:                      # if the follow-up found too little, top up to k
                if len(updated) >= k:
                    break
                if c["id"] not in {u["id"] for u in updated}:
                    updated.append(c)
            step["added"] = [c["id"] for c in added]
            step["dropped"] = [c["id"] for c in chunks if c["id"] not in {u["id"] for u in updated}]
            chunks = updated
        self.last_trace = trace
        return chunks

    def ask(self, question):
        result = super().ask(question)
        result["trace"] = self.last_trace
        return result


# ---------------- Day 5: Agentic RAG ----------------
AGENT_MODEL = "gemini-3.1-flash-lite"   # main agent (Gemini, 250K tokens/min) - use the exact name from AI Studio
FALLBACK_MODEL = "openai/gpt-oss-20b"   # used only if Gemini fails or hits a limit (Groq)
MAX_SEARCHES = 4                        # step limit: at most 4 searches per question
SEARCH_K = 5                            # results returned by one search
GEMINI_PAUSE = 4.5                      # seconds after each Gemini call (15 requests/minute)
GROQ_PAUSE = 20                         # seconds after each fallback call (8,000 tokens/minute)
FALLBACK_WORDS = 100                    # the fallback sees shorter passages, to fit Groq's token limit

AGENT_PROMPT = """You are a research agent for football rulebooks (IFAB Laws of the Game,
UEFA Champions League Regulations, UEFA Disciplinary Regulations, FIFA Regulations on the Status and Transfer of Players).

You have one tool: search(query) -> the 5 most relevant passages from the rulebooks.
Your job is NOT to answer. Your job is to collect passages that together cover EVERY part of the question,
then select the best ones for the writer who will answer.

How to work:
- If the question has several parts (e.g. "what happens on the pitch AND what ban follows"),
  search for each part separately.
- Write short, specific queries. Mention the rulebook when you can guess it.
- If a search did not find what you need, try a different query.
- You can search at most {max_searches} times. Stop as soon as every part is covered.
- If the rulebooks cannot answer the question, finish with the closest passages.

Reply with JSON only, one of:
{"thought": "<short reasoning>", "action": "search", "query": "<search query>"}
{"thought": "<short reasoning>", "action": "finish", "selected": ["<passage id>", ...]}   (at most {k} ids)

Question: {question}

{history}
{instruction}"""


class AgenticRAG(HybridRerankRAG):
    """Day 5: an agent (Gemini 3.1 Flash Lite, fallback gpt-oss-20b) decides what to search, how many times
    (max 4), and which 5 passages to keep. Each search uses the Day 3 search (hybrid + reranking).
    The answer is still written by gpt-oss-120b from 5 plain chunks, as on Days 1-4."""

    def __init__(self):
        from openai import OpenAI
        from tracing import start_tracing
        super().__init__()
        self.gemini_chat = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                                  base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        start_tracing()

    def _history(self, searches, seen, max_words=None):
        if not searches:
            return "Searches so far: none."
        lines, shown = [f"Searches so far ({len(searches)} of {MAX_SEARCHES}):"], set()
        for n, (query, ids) in enumerate(searches, start=1):
            lines.append(f'\nSearch {n}: "{query}"')
            for cid in ids:
                if cid in shown:
                    lines.append(f"[{cid}] (already shown above)")
                    continue
                shown.add(cid)
                text = self.index_text(seen[cid])
                if max_words:
                    text = " ".join(text.split()[:max_words]) + " ..."
                lines.append(f"[{cid}] ({seen[cid]['source']}, p.{seen[cid]['page_start']}) {text}")
        return "\n".join(lines)

    def _prompt(self, question, searches, seen, force, max_words=None):
        instruction = ("You have used all your searches. You MUST finish now and select the passages."
                       if force else "Decide your next action.")
        return (AGENT_PROMPT.replace("{max_searches}", str(MAX_SEARCHES)).replace("{k}", str(TOP_K))
                .replace("{question}", question).replace("{history}", self._history(searches, seen, max_words))
                .replace("{instruction}", instruction))

    @staticmethod
    def _parse(text):
        try:
            return json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        except (AttributeError, json.JSONDecodeError, TypeError):
            return None

    def agent_step(self, question, searches, seen, force):
        """Ask Gemini for the next action; if it fails or hits a limit, ask the fallback instead."""
        try:
            reply = self.gemini_chat.chat.completions.create(
                model=AGENT_MODEL, temperature=0,
                messages=[{"role": "user", "content": self._prompt(question, searches, seen, force)}])
            time.sleep(GEMINI_PAUSE)
            decision = self._parse(reply.choices[0].message.content)
            if decision:
                return decision, "gemini"
        except Exception as err:
            print(f"  agent: Gemini failed ({str(err)[:80]}), switching to fallback")
        prompt = self._prompt(question, searches, seen, force, max_words=FALLBACK_WORDS)
        for attempt in range(3):
            try:
                reply = self.llm.chat.completions.create(
                    model=FALLBACK_MODEL, temperature=0, messages=[{"role": "user", "content": prompt}])
                time.sleep(GROQ_PAUSE)
                return self._parse(reply.choices[0].message.content), "fallback"
            except Exception as err:
                wait = 30 * (attempt + 1)
                print(f"  agent: fallback error ({str(err)[:80]}); retrying in {wait}s...")
                time.sleep(wait)
        return None, "none"

    def retrieve(self, question, k=TOP_K):
        from tracing import set_output, span
        seen, searches, trace, selected = {}, [], [], []
        with span("agent", "AGENT", question) as agent_span:
            for step in range(1, MAX_SEARCHES + 2):
                force = len(searches) >= MAX_SEARCHES
                decision, model = self.agent_step(question, searches, seen, force)
                if not decision:
                    trace.append({"round": step, "note": "agent gave no usable answer", "model": model})
                    break
                if decision.get("action") == "search" and not force and (decision.get("query") or "").strip():
                    query = decision["query"].strip()
                    with span("search_rulebooks", "TOOL", query) as s:
                        results = HybridRerankRAG.retrieve(self, query, SEARCH_K)
                        set_output(s, [c["id"] for c in results])
                    for c in results:
                        seen.setdefault(c["id"], c)
                    searches.append((query, [c["id"] for c in results]))
                    trace.append({"round": step, "action": "search", "query": query,
                                  "thought": decision.get("thought", ""), "model": model})
                    continue
                selected = [cid for cid in decision.get("selected", []) if cid in seen]
                selected = list(dict.fromkeys(selected))[:k]
                trace.append({"round": step, "action": "finish", "selected": list(selected),
                              "thought": decision.get("thought", ""), "model": model})
                break

            if not searches:   # the agent never searched: fall back to one Day 3 search
                results = HybridRerankRAG.retrieve(self, question, SEARCH_K)
                for c in results:
                    seen.setdefault(c["id"], c)
                searches.append((question, [c["id"] for c in results]))
            # always give the writer k chunks: top up with the best results of each search, in turn
            for rank in range(SEARCH_K):
                for _, ids in searches:
                    if len(selected) < k and rank < len(ids) and ids[rank] not in selected:
                        selected.append(ids[rank])
            set_output(agent_span, selected)
        self.last_trace = trace
        return [seen[cid] for cid in selected[:k]]

    def ask(self, question):
        from tracing import set_output, span
        with span("rag_question", "CHAIN", question) as s:
            result = super().ask(question)
            set_output(s, result["answer"])
        result["trace"] = self.last_trace
        return result


AGENT_PROMPT_V2 = """You are a research agent for football rulebooks (IFAB Laws of the Game,
UEFA Champions League Regulations, UEFA Disciplinary Regulations, FIFA Regulations on the Status and Transfer of Players).

You have one tool: search(query) -> the 5 most relevant passages from the rulebooks.
Your job is NOT to answer. Your job is to collect passages that together cover EVERY part of the question,
then select the best ones for the writer who will answer.

Rules:
1. Split the question into its parts (sub-questions). A simple question has 1 part.
   Example: "What happens on the pitch, and is he banned?" has 2 parts.
2. Search 1 below was already run with the full question.
3. If the question has 2 or more parts, EACH part needs its OWN search with a short query for that part only.
   Never combine two parts in one query.
4. Mention the rulebook in the query when you can guess it.
5. At most {max_searches} searches in total (search 1 included). Finish when every part is covered.

Reply with JSON only, one of:
{"thought": "<short>", "parts": ["<part 1>", ...], "action": "search", "part": <part number>, "query": "<query>"}
{"thought": "<short>", "parts": ["<part 1>", ...], "action": "finish", "selected": ["<passage id>", ...]}
Select at most {k} passages, covering every part.

Question: {question}
{parts}
{history}
{instruction}"""


class AgenticRAGv2(AgenticRAG):
    """Day 5, second version. Two general fixes after the first run:
    1) Search the original question first (Day 3 search), so the agent can never start from a worse query.
    2) Plan first: the agent lists the parts of the question; each part of a multi-part question gets its own
       search. If the agent tries to finish early, the uncovered parts are searched automatically."""

    def _prompt(self, question, searches, seen, force, max_words=None, instruction=None):
        parts = ("Parts you listed: " + " | ".join(f"{i}. {p}" for i, p in enumerate(self._parts, start=1))
                 if self._parts else "")
        if instruction is None:
            instruction = ("You have used all your searches. You MUST finish now and select the passages."
                           if force else "Decide your next action.")
        return (AGENT_PROMPT_V2.replace("{max_searches}", str(MAX_SEARCHES)).replace("{k}", str(TOP_K))
                .replace("{question}", question).replace("{parts}", parts)
                .replace("{history}", self._history(searches, seen, max_words)).replace("{instruction}", instruction))

    def _search(self, query, seen, searches):
        from tracing import set_output, span
        with span("search_rulebooks", "TOOL", query) as s:
            results = HybridRerankRAG.retrieve(self, query, SEARCH_K)
            set_output(s, [c["id"] for c in results])
        for c in results:
            seen.setdefault(c["id"], c)
        searches.append((query, [c["id"] for c in results]))

    def retrieve(self, question, k=TOP_K):
        from tracing import set_output, span
        self._parts = None
        seen, searches, trace, selected, covered = {}, [], [], [], set()
        auto_done = False
        with span("agent", "AGENT", question) as agent_span:
            self._search(question, seen, searches)                      # fix 1: original question first
            trace.append({"round": 0, "action": "search", "query": question, "model": "auto (original question)"})
            for step in range(1, MAX_SEARCHES + 3):
                force = len(searches) >= MAX_SEARCHES
                decision, model = self.agent_step(question, searches, seen, force)
                if not decision:
                    trace.append({"round": step, "note": "agent gave no usable answer", "model": model})
                    break
                if self._parts is None:
                    self._parts = [str(p).strip() for p in decision.get("parts", []) if str(p).strip()] or [question]
                    trace.append({"round": step, "note": f"plan: {len(self._parts)} part(s): "
                                  + " | ".join(self._parts), "model": model})
                query = (decision.get("query") or "").strip()
                if decision.get("action") == "search" and not force and query:
                    self._search(query, seen, searches)
                    if isinstance(decision.get("part"), int):
                        covered.add(decision["part"])
                    trace.append({"round": step, "action": "search", "query": query,
                                  "thought": decision.get("thought", ""), "model": model})
                    continue
                # fix 2: before finishing, every part of a multi-part question needs its own search
                n_parts = len(self._parts)
                uncovered = [] if n_parts == 1 else [i for i in range(1, n_parts + 1) if i not in covered]
                if uncovered and not auto_done and len(searches) < MAX_SEARCHES:
                    for i in uncovered:
                        if len(searches) >= MAX_SEARCHES:
                            break
                        self._search(self._parts[i - 1], seen, searches)
                        covered.add(i)
                        trace.append({"round": step, "action": "search", "query": self._parts[i - 1],
                                      "model": "auto (uncovered part)"})
                    auto_done = True
                    continue
                selected = list(dict.fromkeys(cid for cid in decision.get("selected", []) if cid in seen))[:k]
                trace.append({"round": step, "action": "finish", "selected": list(selected),
                              "thought": decision.get("thought", ""), "model": model})
                break
            for rank in range(SEARCH_K):                                 # always give the writer k chunks
                for _, ids in searches:
                    if len(selected) < k and rank < len(ids) and ids[rank] not in selected:
                        selected.append(ids[rank])
            set_output(agent_span, selected)
        self.last_trace = trace
        return [seen[cid] for cid in selected[:k]]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many substitutes can a team use in a UEFA Champions League match?"
    rag = AgenticRAGv2()
    result = rag.ask(question)

    print(f"\nQUESTION: {question}\n")
    print(f"ANSWER:\n{result['answer']}\n")
    print("RETRIEVED CHUNKS:")
    for i, c in enumerate(result["contexts"], start=1):
        print(f"[{i}] {c['score']:.3f} | {c['source']} | p.{c['page_start']}-{c['page_end']}")
    for step in result.get("trace", []):
        print("CHECK:", step)
