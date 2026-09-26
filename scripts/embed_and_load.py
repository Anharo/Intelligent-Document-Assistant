"""
T08: Generate embeddings for the handbook chunks and load them into Qdrant.

Model: BAAI/bge-large-en-v1.5 (1024-dim). First run downloads ~1.3GB from
Hugging Face -- this can take a few minutes depending on your connection.

Run: uv run python scripts/embed_and_load.py
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
COLLECTION_NAME = "handbook_chunks"
CHUNKS_PATH = Path("data/processed/handbook_chunks.json")


def main():
    print(f"Loading embedding model: {MODEL_NAME} (first run downloads the model, be patient)")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {dim}")

    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    client = QdrantClient(url=os.getenv("QDRANT_URL"))

    # Fresh collection each run -- safe to re-run this script any time
    # the chunks change; it always reflects the latest data/processed/ output.
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}' (dim={dim}, cosine distance)")

    texts = [c["text"] for c in chunks]
    print("Encoding chunks (this is the slow part)...")
    embeddings = model.encode(
        texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True
    )

    points = [
        PointStruct(
            id=i,
            vector=vec.tolist(),
            payload={
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "section": c["section"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "content_type": c["content_type"],
                "text": c["text"],
            },
        )
        for i, (c, vec) in enumerate(zip(chunks, embeddings))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nUpserted successfully. Collection '{COLLECTION_NAME}' now holds {count} points.")


if __name__ == "__main__":
    main()