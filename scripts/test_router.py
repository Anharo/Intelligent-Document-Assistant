"""
DoD check for T16: the router should classify 8+/10 hand-picked queries
correctly. This set has 20 (including the tricky rule-vs-data cases) so you
get a more honest number. Uses only the classifier -- no models or databases
needed, so it's fast to iterate on the prompt.

Run: uv run python scripts/test_router.py
"""
import time

from ida.router.classifier import classify_query

# (question, expected_route)
CASES = [
    # --- DOCUMENT_RAG: policies / procedures / definitions ---
    ("What is the alcohol policy on campus?", "DOCUMENT_RAG"),
    ("How do I request a leave of absence?", "DOCUMENT_RAG"),
    ("What is FERPA?", "DOCUMENT_RAG"),
    ("What happens if I get caught cheating on an exam?", "DOCUMENT_RAG"),
    ("What are the rules about smoking on campus?", "DOCUMENT_RAG"),
    ("What is the minimum attendance required to sit for exams?", "DOCUMENT_RAG"),
    ("How does the student judicial resolution process work?", "DOCUMENT_RAG"),
    # --- SQL_QUERY: counts / lists / aggregates over records ---
    ("How many students have unpaid fees?", "SQL_QUERY"),
    ("List students with unpaid fees who also have a disciplinary record.", "SQL_QUERY"),
    ("What is the average grade points per program?", "SQL_QUERY"),
    ("How many students are currently on leave of absence?", "SQL_QUERY"),
    ("Show the top 5 students by number of disciplinary incidents.", "SQL_QUERY"),
    ("Which students have attendance below 65% in CS101?", "SQL_QUERY"),
    ("Which hostel block has the most active residents?", "SQL_QUERY"),
    # --- DIRECT_ANSWER: no retrieval needed ---
    ("Hi there!", "DIRECT_ANSWER"),
    ("Thanks, that was helpful.", "DIRECT_ANSWER"),
    ("What can you help me with?", "DIRECT_ANSWER"),
    ("What is the capital of France?", "DIRECT_ANSWER"),
    # --- tricky: mentions a policy term but wants actual data ---
    ("Which students are below the minimum attendance requirement?", "SQL_QUERY"),
    ("How many students have been sanctioned for academic dishonesty?", "SQL_QUERY"),
]


def main():
    correct = 0
    misses = []
    for q, expected in CASES:
        result = classify_query(q)
        ok = result["route"] == expected
        correct += ok
        mark = "OK  " if ok else "MISS"
        print(f"{mark} {result['route']:<14} (want {expected:<13}) {q}")
        if not ok:
            misses.append((q, expected, result))
        time.sleep(0.3)  # be gentle with free-tier rate limits

    print(f"\nAccuracy: {correct}/{len(CASES)} = {correct / len(CASES):.0%}")
    for q, expected, result in misses:
        print(f"\n  MISS: {q}\n    expected {expected}, got {result['route']}\n    reason: {result['reason']}")


if __name__ == "__main__":
    main()
