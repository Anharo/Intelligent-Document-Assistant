"""
Quick inspection of the handbook PDF: page count, sample text, and
whether PyMuPDF detects any tables on a handful of sample pages.

Run: uv run python scripts/inspect_pdf.py
"""
import fitz  # pymupdf

PDF_PATH = "data/raw/docs/nsu_student_handbook_2025-26.pdf"

doc = fitz.open(PDF_PATH)
print(f"Total pages: {doc.page_count}\n")
sample_pages = [0, doc.page_count // 4, doc.page_count // 2, (3 * doc.page_count) // 4, doc.page_count - 1]

for pno in sample_pages:
    page = doc[pno]
    text = page.get_text()
    tables = page.find_tables()
    print(f"--- Page {pno + 1} ---")
    print(f"  Text length: {len(text)} chars")
    print(f"  Tables detected: {len(tables.tables)}")
    print(f"  First 150 chars: {text[:150].strip()!r}")
    print()
    
doc.close()
