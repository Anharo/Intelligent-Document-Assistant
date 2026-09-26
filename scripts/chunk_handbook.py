"""
Semantic chunking (T07). Takes data/processed/handbook_parsed.json (raw
page/section units from T06) and produces data/processed/handbook_chunks.json:
consistently-sized chunks (~500-600 tokens, 50-token overlap) ready to embed.

Rules:
- Tables are NEVER split or merged with prose -- each table stays its own chunk.
- Consecutive prose units from the SAME section get merged until they hit the
  target size, then a new chunk starts (with a small overlap carried forward).
- A single unit longer than the max size gets split on sentence boundaries.
- Every chunk keeps its source metadata (page range, section) and gets a
  stable chunk_id.

Run: uv run python scripts/chunk_handbook.py
"""
import json
import re
from pathlib import Path

IN_PATH = Path("data/processed/handbook_parsed.json")
OUT_PATH = Path("data/processed/handbook_chunks.json")

# Approximate "tokens" by word count (1 token =~ 0.75 words is the common
# rule of thumb, so target ~550 tokens =~ 410 words). Good enough for
# chunk sizing -- we don't need exact tokenizer counts here.
TARGET_WORDS = 410
MAX_WORDS = 490
OVERLAP_WORDS = 35


def n_tokens(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    # Simple sentence splitter; good enough for policy prose.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def overlap_tail(text: str, n_words: int) -> str:
    """Return the last ~n_words words of text, as a string, for overlap."""
    words = text.split()
    tail = words[-n_words:] if len(words) > n_words else words
    return " ".join(tail)


def main():
    with open(IN_PATH) as f:
        units = json.load(f)

    chunks = []
    chunk_id = 0

    # group consecutive units by (source, section) so we never merge across
    # a section boundary
    i = 0
    while i < len(units):
        unit = units[i]

        if unit["content_type"] == "table":
            chunk_id += 1
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:04d}",
                "source": unit["source"],
                "section": unit["section"],
                "page_start": unit["page"],
                "page_end": unit["page"],
                "content_type": "table",
                "text": unit["text"],
                "n_tokens": n_tokens(unit["text"]),
            })
            i += 1
            continue

        # accumulate consecutive prose units within the same section
        buffer_text = ""
        buffer_pages = []
        section = unit["section"]
        source = unit["source"]

        while i < len(units) and units[i]["content_type"] == "prose" and units[i]["section"] == section:
            candidate = (buffer_text + " " + units[i]["text"]).strip() if buffer_text else units[i]["text"]

            if n_tokens(candidate) > MAX_WORDS and buffer_text:
                # flush current buffer before adding this unit
                break

            buffer_text = candidate
            buffer_pages.append(units[i]["page"])

            if n_tokens(buffer_text) >= TARGET_WORDS:
                i += 1
                break
            i += 1

        if not buffer_text:
            i += 1
            continue

        # if a single unit alone exceeds MAX_WORDS, split it by sentences
        if n_tokens(buffer_text) > MAX_WORDS:
            sentences = split_sentences(buffer_text)
            sub_buffer = ""
            for sent in sentences:
                candidate = (sub_buffer + " " + sent).strip() if sub_buffer else sent
                if n_tokens(candidate) > MAX_WORDS and sub_buffer:
                    chunk_id += 1
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_id:04d}",
                        "source": source, "section": section,
                        "page_start": min(buffer_pages), "page_end": max(buffer_pages),
                        "content_type": "prose", "text": sub_buffer,
                        "n_tokens": n_tokens(sub_buffer),
                    })
                    sub_buffer = overlap_tail(sub_buffer, OVERLAP_WORDS) + " " + sent
                else:
                    sub_buffer = candidate
            if sub_buffer:
                chunk_id += 1
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id:04d}",
                    "source": source, "section": section,
                    "page_start": min(buffer_pages), "page_end": max(buffer_pages),
                    "content_type": "prose", "text": sub_buffer,
                    "n_tokens": n_tokens(sub_buffer),
                })
        else:
            chunk_id += 1
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:04d}",
                "source": source, "section": section,
                "page_start": min(buffer_pages), "page_end": max(buffer_pages),
                "content_type": "prose", "text": buffer_text,
                "n_tokens": n_tokens(buffer_text),
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Safety net: merge any leftover tiny prose chunk (e.g. an orphaned
    # sub-heading label with no real body) into a neighbor in the same
    # section -- try the following chunk first (common case: a short label
    # like "Premeeting" belongs with the paragraph right after it), then
    # fall back to the preceding chunk. A tiny chunk with no same-section
    # neighbor at all (rare) is left as-is.
    MIN_WORDS = 20
    merged = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if c["content_type"] == "prose" and c["n_tokens"] < MIN_WORDS:
            nxt = chunks[i + 1] if i + 1 < len(chunks) else None
            if nxt and nxt["content_type"] == "prose" and nxt["section"] == c["section"]:
                nxt["text"] = c["text"] + " " + nxt["text"]
                nxt["page_start"] = min(c["page_start"], nxt["page_start"])
                nxt["n_tokens"] = n_tokens(nxt["text"])
                i += 1
                continue
            if merged and merged[-1]["content_type"] == "prose" and merged[-1]["section"] == c["section"]:
                merged[-1]["text"] = merged[-1]["text"] + " " + c["text"]
                merged[-1]["page_end"] = max(merged[-1]["page_end"], c["page_end"])
                merged[-1]["n_tokens"] = n_tokens(merged[-1]["text"])
                i += 1
                continue
        merged.append(c)
        i += 1
    chunks = merged

    # re-sequence chunk_ids after merging (some indices were consumed above)
    for idx, c in enumerate(chunks, start=1):
        c["chunk_id"] = f"chunk_{idx:04d}"

    with open(OUT_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    token_counts = [c["n_tokens"] for c in chunks]
    print(f"Created {len(chunks)} chunks from {len(units)} raw units")
    print(f"  Avg tokens/chunk: {sum(token_counts)/len(token_counts):.0f}")
    print(f"  Min: {min(token_counts)}  Max: {max(token_counts)}")
    print(f"  Table chunks: {sum(1 for c in chunks if c['content_type']=='table')}")
    print(f"  Output: {OUT_PATH}")
    print("\nSample chunk:")
    if chunks:
        sample_idx = min(5, len(chunks) - 1)
        print(json.dumps(chunks[sample_idx], indent=2)[:600])


if __name__ == "__main__":
    main()