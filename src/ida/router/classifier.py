"""
T16 (part 1): Prompt-based query router.

Classifies a user question into one of three routes:

  DOCUMENT_RAG   -> answer from the student handbook (policies, rules, procedures)
  SQL_QUERY      -> answer from the student-records database (counts, lists, aggregates)
  DIRECT_ANSWER  -> no retrieval needed (greetings, chit-chat, general knowledge)

Deliberately has no heavy imports (no embedding models, no Qdrant) so you can
iterate on the prompt quickly with scripts/test_router.py.
"""
import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

ROUTES = ("DOCUMENT_RAG", "SQL_QUERY", "DIRECT_ANSWER")
DEFAULT_ROUTE = "DOCUMENT_RAG"  # safest fallback: retrieval can at least say "not found"

SYSTEM_PROMPT = """You are the query router for a university student-affairs assistant. \
The assistant has two knowledge sources:

1. HANDBOOK (documents): the student handbook -- policies, rules, procedures, \
definitions, rights and responsibilities (e.g. alcohol policy, academic integrity, \
leave of absence process, FERPA, judicial/disciplinary process, smoking rules, \
minimum attendance requirement as a RULE).
2. RECORDS DATABASE (SQL): per-student data in tables students, attendance, grades, \
fees, hostel, disciplinary_records (enrollment status, attendance, grades/GPA, fee \
dues, hostel allotment, disciplinary incidents).

Pick exactly ONE route:

- DOCUMENT_RAG: the user asks what a policy/rule/procedure says, how a process works, \
what something means, or what the rules or consequences are. The answer lives in prose.
- SQL_QUERY: the user asks to count, list, filter, rank, average, or look up actual \
records about students (numbers, names, who/how many/which students). The answer \
must be computed from the database.
- DIRECT_ANSWER: greetings, thanks, questions about what the assistant can do, or \
general-knowledge questions unrelated to the handbook or the records.

Tricky cases:
- A question about a RULE is DOCUMENT_RAG even if it mentions a number or table topic \
("What is the minimum attendance required?" -> DOCUMENT_RAG).
- A question about ACTUAL STUDENT DATA is SQL_QUERY even if it mentions a policy term \
("Which students are below the minimum attendance?" -> SQL_QUERY).
- If a question needs both, pick the route that answers the main thing being asked.

Examples:
Q: What is the alcohol policy on campus? -> DOCUMENT_RAG
Q: How do I request a leave of absence? -> DOCUMENT_RAG
Q: What happens if a student is caught cheating? -> DOCUMENT_RAG
Q: How many students have unpaid fees? -> SQL_QUERY
Q: List students in Block A of the hostel. -> SQL_QUERY
Q: What is the average grade points per program? -> SQL_QUERY
Q: Hi there! -> DIRECT_ANSWER
Q: What can you help me with? -> DIRECT_ANSWER
Q: What is the capital of France? -> DIRECT_ANSWER

Respond with ONLY a JSON object, no markdown, in this exact form:
{"route": "<DOCUMENT_RAG|SQL_QUERY|DIRECT_ANSWER>", "reason": "<one short sentence>"}"""


def _parse(text: str) -> tuple[str, str]:
    """Pull (route, reason) out of the model output, tolerating stray text/fences."""
    text = (text or "").strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            route = str(data.get("route", "")).strip().upper()
            if route in ROUTES:
                return route, str(data.get("reason", "")).strip()
        except json.JSONDecodeError:
            pass

    # Fallback: the first route label that appears anywhere in the text.
    for route in ROUTES:
        if route in text.upper():
            return route, "(parsed from unstructured output)"

    return DEFAULT_ROUTE, "(unparseable router output; used default route)"


def classify_query(question: str) -> dict:
    """
    Returns {route, reason}. Never raises on model/parse trouble: falls back
    to DEFAULT_ROUTE so one bad response can't crash the whole /query flow.
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=600,  # headroom: reasoning models spend tokens before answering
            temperature=0,
        )
        route, reason = _parse(response.choices[0].message.content)
    except Exception as e:  # network/rate-limit/etc.
        return {"route": DEFAULT_ROUTE, "reason": f"(router call failed: {e})"}

    return {"route": route, "reason": reason}
