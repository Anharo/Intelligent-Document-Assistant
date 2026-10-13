"""
DoD check for T14/T15: compare the top-3 BEFORE re-ranking (RRF order) and
AFTER (cross-encoder order) for the same queries. Eyeball whether the AFTER
list is cleaner, and note which chunks moved.

Run: uv run python scripts/test_rerank.py
"""
from ida.retrieval.pipeline import retrieve

TEST_QUERIES = [
    "What is the alcohol policy on campus?",
    "How do I request a leave of absence?",
    "What happens if I get caught cheating on an exam?",
    "What are the rules about smoking on campus?",
    "What happens during a Student Judicial Resolution (SJR)?",
    "What is FERPA?",
]


def main():
    for q in TEST_QUERIES:
        print(f"\n{'='*70}\nQUERY: {q}\n{'='*70}")

        print("  --- BEFORE (hybrid RRF only) ---")
        for i, r in enumerate(retrieve(q, top_k=3, rerank=False), start=1):
            print(f"    #{i}  {r['rrf_score']:.4f}  {r['section']!r} (p.{r['page_start']})")

        print("  --- AFTER (cross-encoder re-ranked) ---")
        for i, r in enumerate(retrieve(q, top_k=3), start=1):
            print(
                f"    #{i}  final {r['final_score']:.3f}  ce {r['rerank_score']:+.3f}  "
                f"{r['section']!r} (p.{r['page_start']})  "
                f"[rrf #{r['rank_before']}, ce #{r['rank_rerank']}]"
            )


if __name__ == "__main__":
    main()
