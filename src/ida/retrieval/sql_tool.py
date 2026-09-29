"""
T13: NL -> SQL function.

Takes a natural-language question, has the LLM generate a SQL query against
the known student-records schema, validates it's read-only (SELECT only --
never trust generated SQL to self-police), executes it against Postgres,
and returns the rows plus the SQL itself for transparency.
"""
import os
import re

import psycopg
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SCHEMA_DESCRIPTION = """
Tables (Postgres):

students(student_id VARCHAR PK, name VARCHAR, program VARCHAR, year SMALLINT,
         enrollment_status VARCHAR, email VARCHAR)
  -- enrollment_status one of: active, leave_of_absence, withdrawn

attendance(id SERIAL PK, student_id VARCHAR FK->students, course_code VARCHAR,
           month VARCHAR '2025-07' style, classes_held SMALLINT, classes_attended SMALLINT)

grades(id SERIAL PK, student_id VARCHAR FK->students, course_code VARCHAR,
       semester VARCHAR, credits SMALLINT, grade VARCHAR, grade_points SMALLINT)

fees(id SERIAL PK, student_id VARCHAR FK->students, semester VARCHAR,
     amount_due INTEGER, amount_paid INTEGER, due_date DATE, status VARCHAR)
  -- status one of: paid, partial, unpaid

hostel(id SERIAL PK, student_id VARCHAR FK->students, block VARCHAR,
       room_no VARCHAR, allotted_date DATE, status VARCHAR)
  -- status one of: active, vacated

disciplinary_records(id SERIAL PK, student_id VARCHAR FK->students,
       incident_date DATE, category VARCHAR, description TEXT, sanction VARCHAR)
"""

SYSTEM_PROMPT = f"""You translate natural-language questions into a single \
read-only PostgreSQL SELECT query against this schema:

{SCHEMA_DESCRIPTION}

Rules:
- Output ONLY the SQL query. No explanation, no markdown code fences, no commentary.
- The query MUST start with SELECT.
- Never write INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or GRANT.
- Use explicit column names and appropriate JOINs on student_id.
- For percentage/rate questions, compute from classes_attended / classes_held.
- Always LIMIT results to 100 unless the question asks for a single aggregate value.
"""

FORBIDDEN_KEYWORDS = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant"]


class UnsafeSQLError(Exception):
    pass


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_sql(question: str) -> str:
    """Ask the LLM to translate the question into a SQL query."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=400,
        temperature=0,
    )
    sql = _strip_code_fences(response.choices[0].message.content)
    return sql


def validate_sql(sql: str) -> None:
    """Raises UnsafeSQLError if the query isn't a safe read-only SELECT."""
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        raise UnsafeSQLError(f"Query must start with SELECT. Got: {sql[:50]!r}")
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", normalized):
            raise UnsafeSQLError(f"Forbidden keyword {kw!r} found in generated query.")
    if ";" in sql.rstrip(";").rstrip():
        # a semicolon anywhere but the very end suggests a stacked/second statement
        raise UnsafeSQLError("Multiple statements are not allowed.")


def execute_sql(sql: str) -> list[dict]:
    dsn = os.getenv("POSTGRES_DSN")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def nl_to_sql_query(question: str) -> dict:
    """
    Full pipeline: question -> SQL -> validated -> executed -> rows.

    Returns: {question, sql, rows, error (if any)}
    """
    sql = generate_sql(question)
    try:
        validate_sql(sql)
    except UnsafeSQLError as e:
        return {"question": question, "sql": sql, "rows": None, "error": str(e)}

    try:
        rows = execute_sql(sql)
    except Exception as e:
        return {"question": question, "sql": sql, "rows": None, "error": f"Execution failed: {e}"}

    return {"question": question, "sql": sql, "rows": rows, "error": None}


if __name__ == "__main__":
    result = nl_to_sql_query("How many students have unpaid fees?")
    print("SQL:", result["sql"])
    print("Rows:", result["rows"])
    print("Error:", result["error"])