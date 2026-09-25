"""Step 2: read the PDFs, clean the text, and split it into chunks.

Run from the project root:  python src/ingest.py
Output: data/chunks.json
"""
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf  # PyMuPDF

DATA_DIR = Path("data")
OUT_FILE = DATA_DIR / "chunks.json"
CHUNK_WORDS = 300     # size of each chunk
OVERLAP_WORDS = 50    # words shared between neighbouring chunks, so no idea is cut in half

DOCS = {
    "laws_of_the_game_2026_27.pdf": "IFAB Laws of the Game 2026/27",
    "ucl_regulations_2026_27.pdf": "UEFA Champions League Regulations 2026/27",
    "uefa_disciplinary_regulations.pdf": "UEFA Disciplinary Regulations",
    "fifa_rstp.pdf": "FIFA Regulations on the Status and Transfer of Players",
}


def read_pages(path):
    with pymupdf.open(path) as pdf:
        return [page.get_text("text") for page in pdf]


def find_repeated_lines(pages, min_share=0.5):
    """Lines that appear on most pages are headers or footers."""
    counts = Counter()
    for text in pages:
        counts.update({l.strip() for l in text.splitlines() if l.strip()})
    limit = max(3, int(len(pages) * min_share))
    return {line for line, n in counts.items() if n >= limit}


def clean_page(text, repeated):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if l not in repeated]
    # Drop page numbers (a lone number as the first or last line of the page)
    if lines and re.fullmatch(r"\d{1,4}", lines[-1]):
        lines = lines[:-1]
    if lines and re.fullmatch(r"\d{1,4}", lines[0]):
        lines = lines[1:]
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words split across lines
    return re.sub(r"\s+", " ", text).strip()


def chunk_document(pages, file_name, title):
    words = []  # each word keeps its page number, for citations later
    for page_no, text in enumerate(pages, start=1):
        words += [(w, page_no) for w in text.split()]

    chunks, step = [], CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_WORDS]
        chunks.append({
            "id": f"{Path(file_name).stem}_{len(chunks):04d}",
            "source": title,
            "page_start": piece[0][1],
            "page_end": piece[-1][1],
            "text": " ".join(w for w, _ in piece),
        })
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def main():
    all_chunks = []
    for file_name, title in DOCS.items():
        path = DATA_DIR / file_name
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        raw = read_pages(path)
        repeated = find_repeated_lines(raw)
        pages = [clean_page(p, repeated) for p in raw]
        chunks = chunk_document(pages, file_name, title)
        all_chunks += chunks
        n_words = sum(len(p.split()) for p in pages)
        print(f"{title}: {len(raw)} pages, {n_words} words, {len(chunks)} chunks")

    OUT_FILE.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal: {len(all_chunks)} chunks saved to {OUT_FILE}")

    if all_chunks:
        sample = all_chunks[len(all_chunks) // 3]
        print(f"\nSample chunk ({sample['source']}, p.{sample['page_start']}-{sample['page_end']}):")
        print(sample["text"][:800])


if __name__ == "__main__":
    main()
