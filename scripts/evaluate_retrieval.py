
"""
Evaluate Dense, BM25, Hybrid RRF, and Hybrid + Cross-Encoder retrieval.

Run:
    uv run python scripts/evaluate_retrieval.py
"""

import json
import math
from pathlib import Path

from ida.retrieval.dense import dense_retrieve
from ida.retrieval.sparse import bm25_retrieve
from ida.retrieval.fusion import rrf_fuse
from ida.retrieval.pipeline import retrieve

GROUND_TRUTH_PATH = Path("data/eval/ground_truth.json")
OUTPUT_PATH = Path("data/eval/results.json")
TOP_K = 5


def recall_at_k(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:TOP_K]) & relevant_ids)
    return hits / len(relevant_ids)


def hit_rate_at_k(retrieved_ids, relevant_ids):
    return float(bool(set(retrieved_ids[:TOP_K]) & relevant_ids))


def reciprocal_rank_at_k(retrieved_ids, relevant_ids):
    for rank, chunk_id in enumerate(retrieved_ids[:TOP_K], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids):
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids[:TOP_K], start=1)
        if chunk_id in relevant_ids
    )

    ideal_hits = min(len(relevant_ids), TOP_K)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def validate_test_cases(test_cases):
    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError("Ground-truth dataset must be a non-empty JSON list.")

    for index, case in enumerate(test_cases):
        if not isinstance(case, dict):
            raise ValueError(f"Test case {index} must be an object.")

        query = case.get("query")
        relevant = case.get("relevant_chunk_ids")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Test case {index} has an invalid query.")

        if not isinstance(relevant, list) or not relevant:
            raise ValueError(
                f"Test case {index} must have non-empty relevant_chunk_ids."
            )

        if not all(isinstance(cid, str) and cid.strip() for cid in relevant):
            raise ValueError(
                f"Test case {index} contains an invalid chunk ID."
            )


def evaluate_config(name, retrieval_function, test_cases):
    metric_names = ["Recall@5", "Hit Rate@5", "MRR@5", "nDCG@5"]

    per_query = []
    metric_totals = {metric: 0.0 for metric in metric_names}

    print(f"\n{'=' * 80}")
    print(f"CONFIGURATION: {name}")
    print("=" * 80)

    for case in test_cases:
        query = case["query"]
        relevant = set(case["relevant_chunk_ids"])

        results = retrieval_function(query)
        retrieved = [r["chunk_id"] for r in results]

        scores = {
            "Recall@5": recall_at_k(retrieved, relevant),
            "Hit Rate@5": hit_rate_at_k(retrieved, relevant),
            "MRR@5": reciprocal_rank_at_k(retrieved, relevant),
            "nDCG@5": ndcg_at_k(retrieved, relevant),
        }

        for metric, value in scores.items():
            metric_totals[metric] += value

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved[:TOP_K], start=1)
                if chunk_id in relevant
            ),
            None,
        )

        per_query.append({
            "query": query,
            "relevant_chunk_ids": sorted(relevant),
            "retrieved_chunk_ids": retrieved[:TOP_K],
            "first_relevant_rank": first_relevant_rank,
            "metrics": scores,
        })

        print(f"\nQuery: {query}")
        print(f"Relevant: {sorted(relevant)}")
        print(f"Retrieved: {retrieved[:TOP_K]}")
        print(f"First relevant rank: {first_relevant_rank or 'Not found'}")

    averages = {
        metric: total / len(test_cases)
        for metric, total in metric_totals.items()
    }

    print(f"\n--- {name} AVERAGES ---")
    for metric, value in averages.items():
        print(f"{metric:15}: {value:.4f} ({value * 100:.2f}%)")

    return {
        "averages": averages,
        "per_query": per_query,
    }


def main():
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Missing {GROUND_TRUTH_PATH}")

    test_cases = json.loads(GROUND_TRUTH_PATH.read_text())
    validate_test_cases(test_cases)

    configurations = [
        ("Dense only", lambda q: dense_retrieve(q, top_k=TOP_K)),
        ("BM25 only", lambda q: bm25_retrieve(q, top_k=TOP_K)),
        ("Hybrid RRF", lambda q: rrf_fuse(q, top_k=TOP_K)),
        ("Hybrid + Reranker", lambda q: retrieve(q, top_k=TOP_K)),
    ]

    results = {}

    for name, function in configurations:
        results[name] = evaluate_config(name, function, test_cases)

    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON")
    print("=" * 80)

    metrics = ["Recall@5", "Hit Rate@5", "MRR@5", "nDCG@5"]

    print(f"{'Configuration':25}", end="")
    for metric in metrics:
        print(f"{metric:>15}", end="")
    print()

    for name, data in results.items():
        print(f"{name:25}", end="")
        for metric in metrics:
            print(f"{data['averages'][metric]:>15.4f}", end="")
        print()

    output = {
        "top_k": TOP_K,
        "query_count": len(test_cases),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
