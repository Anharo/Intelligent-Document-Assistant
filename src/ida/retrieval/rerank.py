"""
T14: Cross-encoder re-ranking.

Hybrid retrieval (dense + BM25 + RRF) is good at RECALL: the right chunk is
usually somewhere in the top ~10. A cross-encoder reads (query, chunk)
together and can sharpen PRECISION -- but on a small, uniform corpus like a
student handbook, bge-reranker-base occasionally gives short, off-topic chunks
a near-perfect score (seen in testing: a 66-word "Noise" policy scored 1.000
for a smoking question, above the actual smoking policy).

So this module does NOT let the re-ranker overrule retrieval outright:

  1. The heading is prepended to each passage, so the model sees the topic.
  2. The final order blends two rankings -- the original RRF rank and the
     cross-encoder rank -- with a rank-fusion formula. A chunk has to be
     good in BOTH to win; a single wild cross-encoder score can't promote a
     chunk that retrieval ranked near the bottom.

Model: BAAI/bge-reranker-base (~1.1GB, first run downloads it).
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# Rank-fusion constant for the blend. Small k => rank position matters a lot,
# which is what we want (RRF's usual k=60 would barely separate ranks 1 and 8).
BLEND_K = 10


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Load the cross-encoder once and reuse it across calls."""
    return CrossEncoder(RERANK_MODEL, max_length=512)


def rerank(query: str, candidates: list[dict], top_k: int = 5, blend: bool = True) -> list[dict]:
    """
    Re-scores `candidates` (each must have 'text'; 'section' is used if present)
    and returns the best top_k.

    Added fields on each result:
      rank_before   1-indexed position in the input list (the RRF order)
      rerank_score  raw cross-encoder score
      rank_rerank   1-indexed position by cross-encoder score alone
      final_score   blended score used for the final ordering

    blend=False orders purely by the cross-encoder (useful for comparing).
    """
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, f"{c.get('section', '')}\n{c['text']}".strip()) for c in candidates]
    scores = model.predict(pairs)

    scored = []
    for rank_before, (c, s) in enumerate(zip(candidates, scores), start=1):
        r = dict(c)
        r["rerank_score"] = float(s)
        r["rank_before"] = rank_before
        scored.append(r)

    by_ce = sorted(scored, key=lambda r: r["rerank_score"], reverse=True)
    for rank, r in enumerate(by_ce, start=1):
        r["rank_rerank"] = rank
        r["final_score"] = (
            1.0 / (BLEND_K + r["rank_before"]) + 1.0 / (BLEND_K + rank)
            if blend
            else r["rerank_score"]
        )

    scored.sort(key=lambda r: r["final_score"], reverse=True)
    return scored[:top_k]
