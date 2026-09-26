"""
Sanity-check T08: run a few real questions against the Qdrant collection
and eyeball whether the returned chunks are actually relevant.

Run: uv run python scripts/test_search.py
"""
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
COLLECTION_NAME = "handbook_chunks"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TEST_QUERIES = [
    "What is the alcohol policy on campus?",
    "How do I request a leave of absence?",
    "What happens if I get caught cheating on an exam?",
    "What are the rules about smoking on campus?",
    "How does the student judicial process work?",
]


def main():
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(url=os.getenv("QDRANT_URL"))

    for query in TEST_QUERIES:
        vec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        response = client.query_points(collection_name=COLLECTION_NAME, query=vec.tolist(), limit=3)
        results = response.points

        print(f"\n{'='*70}\nQUERY: {query}\n{'='*70}")
        for r in results:
            p = r.payload
            print(f"  score={r.score:.3f}  section={p['section']!r}  page={p['page_start']}")
            print(f"    {p['text'][:150]}...")


if __name__ == "__main__":
    main()