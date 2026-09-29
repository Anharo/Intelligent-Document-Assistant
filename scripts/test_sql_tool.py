"""
Stress-test T13 with harder questions: joins, percentages, multi-condition
filters, and semantically ambiguous phrasing.

Run: uv run python scripts/test_sql_tool.py
"""
from ida.retrieval.sql_tool import nl_to_sql_query

TEST_QUESTIONS = [
    "Which students have attendance below 65% in course CS101?",
    "List students with unpaid fees who also have a disciplinary record.",
    "What is the average GPA points per program?",
    "How many students are currently on leave of absence?",
    "Show the top 5 students by number of disciplinary incidents.",
]


def main():
    for q in TEST_QUESTIONS:
        print(f"\n{'='*70}\nQUESTION: {q}\n{'='*70}")
        result = nl_to_sql_query(q)
        print("SQL:", result["sql"])
        if result["error"]:
            print("ERROR:", result["error"])
        else:
            rows = result["rows"]
            print(f"Rows returned: {len(rows)}")
            for r in rows[:5]:
                print("  ", r)


if __name__ == "__main__":
    main()