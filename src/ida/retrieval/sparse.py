"""
T11: BM25 sparse retrieval function.

Keyword-based retrieval over the same chunk set used for dense retrieval.
This is the "sparse" half of the hybrid retrieval pair -- catches exact
term/acronym matches (e.g. "FERPA", "SJR") that dense embeddings can
sometimes under-rank, and gets fused with dense results via RRF in T12.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/processed/handbook_chunks.json")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English stopwords -- without this filter, generic connector words
# ("what", "is", "the", "on"...) accumulate small matches across many long
# chunks and can outrank the chunk that's actually topically relevant.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "there", "these",
    "this", "those", "to", "was", "were", "what", "when", "where", "which",
    "who", "will", "with", "you", "your",
}


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


@lru_cache(maxsize=1)
def _load_index():
    """Load chunks once and build the BM25 index once, reused across calls."""
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks


def bm25_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Returns the top_k chunks with the highest BM25 keyword-match score
    against `query`.

    Each result: {chunk_id, score, section, page_start, page_end,
                  content_type, text}
    """
    bm25, chunks = _load_index()
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in ranked_indices:
        c = chunks[i]
        results.append({
            "chunk_id": c["chunk_id"],
            "score": float(scores[i]),
            "section": c["section"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "content_type": c["content_type"],
            "text": c["text"],
        })
    return results


if __name__ == "__main__":
    # Quick manual check when running this file directly
    for r in bm25_retrieve("What is the alcohol policy?", top_k=3):
        print(f"{r['score']:.3f}  {r['section']}  (p.{r['page_start']})")