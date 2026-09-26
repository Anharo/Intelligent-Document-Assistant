import json

with open("data/processed/handbook_chunks.json") as f:
    chunks = json.load(f)

chunks_sorted = sorted(chunks, key=lambda c: c["n_tokens"])

print("=== 10 SMALLEST CHUNKS ===\n")
for c in chunks_sorted[:10]:
    print(f"[{c['chunk_id']}] {c['n_tokens']} words | section: {c['section']!r} | type: {c['content_type']}")
    print(f"  text: {c['text']!r}")
    print()
