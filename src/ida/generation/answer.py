"""
T17 + T18: turn retrieved material into final natural-language answers.

  generate_rag_answer   handbook chunks  -> cited answer          (T17)
  explain_sql_result    SQL rows         -> plain-language answer  (T18)
  generate_direct_answer  no retrieval   -> short reply
"""
import json
import re

from ida.generation.llm import chat

NOT_FOUND = "I couldn't find this in the student handbook."

RAG_SYSTEM = f"""You answer questions about a university student handbook using \
ONLY the numbered excerpts provided.

Rules:
- Use only information that appears in the excerpts. Never use outside knowledge.
- After each claim, cite the excerpt(s) it came from with bracketed numbers, \
like [1] or [2][3].
- If the excerpts do not contain the answer, reply with exactly: {NOT_FOUND}
- Be concise: a short paragraph, or a few bullet points if the answer is a list.
- Do not mention "excerpts" or "context" in your answer; just answer and cite."""

SQL_SYSTEM = """You explain database query results to a university staff member \
in plain language.

Rules:
- State only what the rows show. Never invent numbers, names, or trends.
- Lead with the direct answer (the count, the name, the average...), then add \
brief supporting detail if useful.
- Round decimals sensibly (1-2 places).
- If the results were truncated, say how many rows exist and that only some are listed.
- If several rows tie on the value being ranked, say so.
- Keep it short. Do not repeat the SQL."""

DIRECT_SYSTEM = """You are IDA, the Intelligent Document Assistant for a university's \
student affairs office. You have two capabilities:

1. Search the student handbook -- policies, rules and procedures. Examples: "What is \
the alcohol policy?", "How do I request a leave of absence?".
2. Query the student records database -- counts, lists and averages ACROSS students \
(enrollment, attendance, grades, fees, hostel, disciplinary records). Examples: "How \
many students have unpaid fees?", "What is the average grade points per program?".

You cannot look up a single user's own account, and you have no login or personal data \
about whoever is chatting with you.

The user's message needs no lookup. Reply briefly and in a friendly tone. If they ask \
what you can do, describe the two capabilities above using those examples only. For \
general-knowledge questions, answer briefly. Never claim to have looked anything up."""

MAX_ROWS_TO_LLM = 30


def _normalize_citations(text: str) -> str:
    """
    Models sometimes write citations as 【1】 (full-width) or [1, 2]. Rewrite
    them all to plain [1][2] so they render consistently and can be parsed.
    """
    text = re.sub(r"[【\[](\d+)[】\]]", r"[\1]", text)
    text = re.sub(
        r"\[(\d+(?:\s*,\s*\d+)+)\]",
        lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")),
        text,
    )
    return text


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        pages = (
            f"p.{c['page_start']}"
            if c["page_start"] == c["page_end"]
            else f"p.{c['page_start']}-{c['page_end']}"
        )
        parts.append(f"[{i}] ({c['section']}, {pages})\n{c['text']}")
    return "\n\n".join(parts)


def _citation(n: int, c: dict) -> dict:
    return {
        "n": n,
        "chunk_id": c["chunk_id"],
        "section": c["section"],
        "page_start": c["page_start"],
        "page_end": c["page_end"],
        "text": c["text"],
    }


def generate_rag_answer(question: str, chunks: list[dict]) -> dict:
    """
    Returns {answer, citations}. `citations` holds only the excerpts the answer
    actually cites (falls back to all retrieved chunks if the model cited none
    but still answered; empty if it said the answer isn't in the handbook).
    """
    if not chunks:
        return {"answer": NOT_FOUND, "citations": []}

    user = f"Excerpts:\n\n{_format_context(chunks)}\n\nQuestion: {question}"
    answer = _normalize_citations(chat(RAG_SYSTEM, user))

    if not answer:
        return {"answer": "The model returned an empty answer. Please try again.", "citations": []}

    refs = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(chunks)})
    if refs:
        cited = [_citation(n, chunks[n - 1]) for n in refs]
    elif NOT_FOUND in answer:
        cited = []
    else:
        cited = [_citation(i, c) for i, c in enumerate(chunks, start=1)]

    return {"answer": answer, "citations": cited}


def explain_sql_result(question: str, sql: str | None, rows: list[dict] | None, error: str | None) -> str:
    """Plain-language explanation of a SQL result. Never raises on odd results."""
    if error:
        return f"I couldn't answer that from the records database. ({error})"
    if rows is None:
        return "The query didn't return any result."
    if len(rows) == 0:
        return "No records match that question."

    shown = rows[:MAX_ROWS_TO_LLM]
    user = (
        f"Question: {question}\n"
        f"SQL used: {sql}\n"
        f"Total rows returned: {len(rows)} (showing {len(shown)})\n"
        f"Rows (JSON): {json.dumps(shown, default=str)}"
    )
    try:
        answer = chat(SQL_SYSTEM, user, max_tokens=800)
    except Exception:
        answer = ""

    if answer:
        return answer
    # Fallback if the LLM is unavailable: still give a usable, honest answer.
    if len(rows) == 1 and len(rows[0]) == 1:
        (k, v), = rows[0].items()
        return f"{k}: {v}"
    return f"The query returned {len(rows)} row(s); see the table below."


def generate_direct_answer(question: str) -> str:
    answer = chat(DIRECT_SYSTEM, question, max_tokens=500, temperature=0.3)
    return answer or "Hi! Ask me about the student handbook or the student records."
