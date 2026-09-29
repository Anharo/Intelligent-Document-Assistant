"""
T12: Reciprocal Rank Fusion (RRF).

Combines dense retrieval and BM25 rankings into a single ranked list.
RRF works purely on RANK POSITION, not raw score -- which is exactly why
it's the standard fusion method here: dense cosine scores (0-1) and BM25
scores (unbounded) live on different scales and can't be added directly.

Formula: for each chunk, score = sum over each ranker of 1 / (k + rank),
where rank is 1-indexed position in that ranker's results and k=60 is the
standard RRF constant (dampens the impact of any single high rank).
"""
from ida.retrieval.dense import dense_retrieve
from ida.retrieval.sparse import bm25_retrieve

RRF_K = 60


def rrf_fuse(query: str, top_k: int = 5, candidate_k: int = 20) -> list[dict]:
    """
    Retrieves candidate_k results from each method, fuses them via RRF,
    and returns the top_k fused results.

    Each result: {chunk_id, rrf_score, section, page_start, page_end,
                  content_type, text}
    """
    dense_results = dense_retrieve(query, top_k=candidate_k)
    bm25_results = bm25_retrieve(query, top_k=candidate_k)

    rrf_scores: dict[str, float] = {}
    chunk_meta: dict[str, dict] = {}

    for rank, r in enumerate(dense_results, start=1):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        chunk_meta[cid] = r

    for rank, r in enumerate(bm25_results, start=1):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        chunk_meta.setdefault(cid, r)

    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    results = []
    for cid in ranked_ids:
        r = dict(chunk_meta[cid])
        r.pop("score", None)  # method-specific score isn't meaningful post-fusion
        r["rrf_score"] = rrf_scores[cid]
        results.append(r)
    return results


if __name__ == "__main__":
    for r in rrf_fuse("What is the alcohol policy?", top_k=5):
        print(f"{r['rrf_score']:.4f}  {r['section']}  (p.{r['page_start']})")