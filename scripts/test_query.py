"""
T21: manual end-to-end test of all 3 paths (no API server or UI needed).

Calls run_query() directly -- the same function /query uses -- on 2 questions
per path, and prints route, answer, and sources so you can eyeball quality.
Needs Qdrant + Postgres running and a valid GROQ_API_KEY.

Run: uv run python scripts/test_query.py
"""
import time

from ida.router.graph import run_query

CASES = [
    ("DOCUMENT_RAG", "What is the alcohol policy on campus?"),
    ("DOCUMENT_RAG", "How do I request a leave of absence?"),
    ("DOCUMENT_RAG", "What is the university's policy on space travel?"),  # should say not found
    ("SQL_QUERY", "How many students have unpaid fees?"),
    ("SQL_QUERY", "What is the average grade points per program?"),
    ("SQL_QUERY", "Show the top 5 students by number of disciplinary incidents."),
    ("DIRECT_ANSWER", "Hi there!"),
    ("DIRECT_ANSWER", "What can you help me with?"),
]


def main():
    failures = 0
    for expected, q in CASES:
        print(f"\n{'=' * 72}\nQ: {q}\n{'=' * 72}")
        r = run_query(q)
        ok = r["route"] == expected and not r["error"] and bool(r["answer"])
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} route={r['route']} (want {expected})  {r['elapsed_s']}s")
        print(f"  router reason: {r['route_reason']}")
        if r["error"]:
            print(f"  error: {r['error']}")
        print(f"\n{r['answer']}")
        for c in r["citations"]:
            print(f"  [{c['n']}] {c['section']} (p.{c['page_start']})")
        if r["sql"]:
            print(f"  SQL: {r['sql']}")
        time.sleep(0.5)  # gentle on free-tier rate limits

    print(f"\n{len(CASES) - failures}/{len(CASES)} paths worked")


if __name__ == "__main__":
    main()
