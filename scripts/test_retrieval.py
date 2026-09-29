"""
DoD check for T10 + T11: run dense retrieval and BM25 retrieval
independently on the same queries and compare top-k side by side.

Run: uv run python scripts/test_retrieval.py
"""
from ida.retrieval.dense import dense_retrieve
from ida.retrieval.sparse import bm25_retrieve
from ida.retrieval.fusion import rrf_fuse

TEST_QUERIES = [
    "What is the alcohol policy on campus?",
    "What is FERPA?",
    "What happens during a Student Judicial Resolution (SJR)?",
    "How do I request a leave of absence?",
]


def show(label, results, score_key="score"):
    print(f"  --- {label} ---")
    for r in results:
        print(f"    {r[score_key]:.4f}  {r['section']!r} (p.{r['page_start']})")


def main():
    for query in TEST_QUERIES:
        print(f"\n{'='*70}\nQUERY: {query}\n{'='*70}")
        show("DENSE", dense_retrieve(query, top_k=3))
        show("BM25", bm25_retrieve(query, top_k=3))
        show("RRF FUSED", rrf_fuse(query, top_k=3), score_key="rrf_score")


if __name__ == "__main__":
    main()