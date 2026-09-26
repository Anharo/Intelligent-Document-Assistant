"""
Layout-aware parser for the campus handbook PDF (T06).

Separates prose text from tables (tables -> markdown), detects section
headers via font size, and tags every extracted unit with page + section
metadata. Output feeds directly into T07's chunking step.

Run: uv run python scripts/parse_pdf.py
"""
import json
import re
from pathlib import Path

import fitz  # pymupdf

PDF_PATH = Path("data/raw/docs/nsu_student_handbook_2025-26.pdf")
OUT_PATH = Path("data/processed/handbook_parsed.json")
SOURCE_NAME = "nsu_student_handbook_2025-26"

# A span counts as a "heading" if its font size is meaningfully larger than
# the page's own median body-text size. This is a heuristic, not a rule.
HEADING_SIZE_MARGIN = 1.5

# Running header/footer boilerplate that repeats on nearly every page and
# carries no content -- stripped before a line is ever treated as prose.
BOILERPLATE_PATTERNS = [
    re.compile(r"^\d*\s*Nova Southeastern University Student Handbook\s*$", re.IGNORECASE),
]

# A bare page number on its own line (e.g. "18").
PAGE_NUMBER_LINE = re.compile(r"^\d{1,4}$")
# A page number glued to the start of an otherwise-real line (e.g. "17 *Date varies...").
LEADING_PAGE_NUMBER = re.compile(r"^\d{1,4}\s+")
# A short "Nova Southeastern University <page#>" footer glued to the end of a line.
TRAILING_FOOTER = re.compile(r"\s*Nova Southeastern University\s+\d{1,3}\s*$", re.IGNORECASE)


def is_boilerplate(line_text: str) -> bool:
    if any(p.match(line_text.strip()) for p in BOILERPLATE_PATTERNS):
        return True
    return bool(PAGE_NUMBER_LINE.match(line_text.strip()))


def clean_line(line_text: str) -> str:
    text = TRAILING_FOOTER.sub("", line_text)
    text = LEADING_PAGE_NUMBER.sub("", text)
    return text.strip()


def table_to_markdown(table) -> str:
    rows = table.extract()
    if not rows:
        return ""
    header, *body = rows
    header = [str(c).strip() if c else "" for c in header]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        row = [str(c).strip().replace("\n", " ") if c else "" for c in row]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 10.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def extract_page(page, page_no, current_section):
    records = []

    tables = page.find_tables()
    table_bboxes = [fitz.Rect(t.bbox) for t in tables.tables]

    for t in tables.tables:
        md = table_to_markdown(t)
        if md:
            records.append({
                "source": SOURCE_NAME, "page": page_no, "section": current_section,
                "content_type": "table", "text": md,
            })

    page_dict = page.get_text("dict")
    all_sizes = [
        span["size"]
        for block in page_dict["blocks"] if block.get("type") == 0
        for line in block["lines"] for span in line["spans"]
    ]
    body_size = median(all_sizes)

    prose_lines = []
    last_was_heading = False
    for block in page_dict["blocks"]:
        if block.get("type") != 0:
            continue
        block_rect = fitz.Rect(block["bbox"])
        if any(block_rect.intersects(tb) for tb in table_bboxes):
            continue  # already captured as a table

        for line in block["lines"]:
            line_text = "".join(span["text"] for span in line["spans"]).strip()
            if not line_text or is_boilerplate(line_text):
                continue
            line_text = clean_line(line_text)
            if not line_text:
                continue
            max_span_size = max(span["size"] for span in line["spans"])
            stripped = line_text.strip()
            looks_like_page_number = stripped.replace(".", "").isdigit()
            is_heading = (
                max_span_size >= body_size + HEADING_SIZE_MARGIN
                and 4 <= len(line_text) < 100
                and not looks_like_page_number
            )
            if is_heading:
                if prose_lines:
                    records.append({
                        "source": SOURCE_NAME, "page": page_no, "section": current_section,
                        "content_type": "prose", "text": " ".join(prose_lines),
                    })
                    prose_lines = []
                if last_was_heading:
                    current_section = f"{current_section} {line_text}"
                else:
                    current_section = line_text
                last_was_heading = True
            else:
                prose_lines.append(line_text)
                last_was_heading = False

    if prose_lines:
        records.append({
            "source": SOURCE_NAME, "page": page_no, "section": current_section,
            "content_type": "prose", "text": " ".join(prose_lines),
        })

    return records, current_section


def main():
    doc = fitz.open(PDF_PATH)
    all_records = []
    current_section = "Front Matter"

    for i, page in enumerate(doc):
        page_no = i + 1
        records, current_section = extract_page(page, page_no, current_section)
        all_records.extend(records)

    doc.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_records, f, indent=2)

    n_prose = sum(1 for r in all_records if r["content_type"] == "prose")
    n_table = sum(1 for r in all_records if r["content_type"] == "table")
    sections = sorted(set(r["section"] for r in all_records))

    print(f"Parsed {len(all_records)} units from {PDF_PATH.name}")
    print(f"  Prose units: {n_prose}")
    print(f"  Table units: {n_table}")
    print(f"  Distinct sections detected: {len(sections)}")
    print(f"  Output: {OUT_PATH}")
    print("\nFirst 5 detected section names:")
    for s in sections[:5]:
        print(f"  - {s}")


if __name__ == "__main__":
    main()