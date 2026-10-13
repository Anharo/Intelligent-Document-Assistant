import os
from functools import lru_cache
 
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
 
load_dotenv()
 
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
COLLECTION_NAME = "handbook_chunks"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
 
 
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it across calls."""
    return SentenceTransformer(MODEL_NAME)
 
 
@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    return QdrantClient(url=os.getenv("QDRANT_URL"), check_compatibility=False)
 
 
def dense_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Returns the top_k chunks most semantically similar to `query`.
 
    Each result: {chunk_id, score, section, page_start, page_end,
                  content_type, text}
    """
    model = _get_model()
    client = _get_client()
 
    vec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    response = client.query_points(
        collection_name=COLLECTION_NAME, query=vec.tolist(), limit=top_k
    )
 
    results = []
    for point in response.points:
        p = point.payload
        results.append({
            "chunk_id": p["chunk_id"],
            "score": point.score,
            "section": p["section"],
            "page_start": p["page_start"],
            "page_end": p["page_end"],
            "content_type": p["content_type"],
            "text": p["text"],
        })
    return results
 
 
if __name__ == "__main__":
    # Quick manual check when running this file directly
    for r in dense_retrieve("What is the alcohol policy?", top_k=3):
        print(f"{r['score']:.3f}  {r['section']}  (p.{r['page_start']})")
 