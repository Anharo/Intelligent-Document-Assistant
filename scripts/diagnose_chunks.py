"""
Diagnose WHY the re-ranker surfaced 'B.29 Noise', 'Table of Contents', etc.

Checks:
  1. How big the chunks are, and the largest ones.
  2. Chunks whose text contains several numbered headings (B.30, B.31, ...)
     -- a sign that section detection missed headings and the chunk swallowed
     multiple policies under one label.
  3. Table-of-Contents chunks (should not be indexed).
  4. For each test query, the top-3 reranked chunks with word count, how many
     headings they contain, and a snippet around the first keyword hit.

Run: uv run python scripts/diagnose_chunks.py
"""
import json
import re
from pathlib import Path

CHUNKS_PATH = Path("data/processed/handbook_chunks.json")
HEADING_RE = re.compile(r"\b[A-D]\.\d{1,2}\s+[A-Z][A-Za-z/&\-\s]{2,40}")

QUERIES = {
    "What is the alcohol policy on campus?": "alcohol",
    "What are the rules about smoking on campus?": "smok",
    "What happens if I get caught cheating on an exam?": "cheat",
    "What is FERPA?": "ferpa",
}


def words(c):
    return len(c["text"].split())


def snippet(text, kw, width=140):
    i = text.lower().find(kw)
    if i < 0:
        return "(keyword not in chunk text)"
    return "..." + text[max(0, i - width): i + width].replace("\n", " ") + "..."


def main():
    chunks = json.loads(CHUNKS_PATH.read_text())
    by_id = {c["chunk_id"]: c for c in chunks}
    sizes = sorted(words(c) for c in chunks)
    print(f"{len(chunks)} chunks | words: min {sizes[0]}, median {sizes[len(sizes)//2]}, max {sizes[-1]}")

    print("\n--- 5 largest chunks ---")
    for c in sorted(chunks, key=words, reverse=True)[:5]:
        print(f"  {words(c):>5} words  {c['section']!r} (p.{c['page_start']}-{c['page_end']})")

    print("\n--- chunks containing 3+ numbered headings (possible swallowed sections) ---")
    suspects = [c for c in chunks if len(HEADING_RE.findall(c["text"])) >= 3]
    for c in suspects[:15]:
        print(f"  {len(HEADING_RE.findall(c['text'])):>2} headings  {c['section']!r} (p.{c['page_start']}-{c['page_end']})")
    print(f"  total suspects: {len(suspects)}")

    print("\n--- Table of Contents chunks ---")
    toc = [c for c in chunks if "table of contents" in c["section"].lower()]
    for c in toc:
        print(f"  {c['chunk_id']}  p.{c['page_start']}-{c['page_end']}  {words(c)} words")
    print(f"  total: {len(toc)}")

    print("\n--- top-3 reranked chunks per query ---")
    from ida.retrieval.pipeline import retrieve

    for q, kw in QUERIES.items():
        print(f"\nQ: {q}")
        for r in retrieve(q, top_k=3):
            c = by_id.get(r["chunk_id"], r)
            n_head = len(HEADING_RE.findall(c["text"]))
            print(f"  {r['rerank_score']:+.3f}  {r['section']!r}  {words(c)} words, {n_head} headings")
            print(f"     {snippet(c['text'], kw)}")


if __name__ == "__main__":
    main()
