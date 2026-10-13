"""
T15: One callable for the whole document-retrieval path.

    retrieve(query) -> dense + BM25 -> RRF fusion -> cross-encoder re-rank

This is the function the router's DOCUMENT_RAG branch (and later the /query
endpoint and the eval harness) will call. `rerank=False` gives you the
"baseline" hybrid config for the Week 7 ablation, so the same function serves
both configurations.
"""
from ida.retrieval.fusion import rrf_fuse
from ida.retrieval.rerank import rerank as _rerank


def retrieve(
    query: str,
    top_k: int = 5,
    pool_size: int = 8,
    candidate_k: int = 20,
    rerank: bool = True,
) -> list[dict]:
    """
    Args:
        query: the user's question.
        top_k: how many chunks to return.
        pool_size: how many fused chunks go into the re-ranker (wider = better
            recall, slower). Ignored when rerank=False.
        candidate_k: how many results each of dense/BM25 contributes to fusion.
        rerank: set False for the baseline (hybrid-only) ablation config.

    Returns chunks as dicts with chunk_id, section, page_start, page_end,
    content_type, text, rrf_score, and (if reranked) rerank_score + rank_before.
    """
    if not rerank:
        return rrf_fuse(query, top_k=top_k, candidate_k=candidate_k)

    pool = rrf_fuse(query, top_k=max(pool_size, top_k), candidate_k=candidate_k)
    return _rerank(query, pool, top_k=top_k)


if __name__ == "__main__":
    for r in retrieve("What is the alcohol policy?"):
        print(f"{r['rerank_score']:.3f}  was #{r['rank_before']}  {r['section']}  (p.{r['page_start']})")
