"""
LangGraph state graph: router -> path -> generation (T16 + T19).

    START -> classify -> (conditional) -> document_rag | sql_query | direct_answer -> END

Each branch node does its own retrieval AND generation, so a single
`run_query(question)` call returns the final answer for any of the three paths.

Heavy imports (embedding model, reranker, Postgres) happen lazily INSIDE the
nodes, so routing on its own stays fast to import and test. Every branch
catches its own failures and returns a readable message instead of raising,
so one flaky dependency can't take the whole demo down.
"""
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from ida.router.classifier import classify_query


class QueryState(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    chunks: list[dict]          # DOCUMENT_RAG: reranked chunks
    citations: list[dict]       # DOCUMENT_RAG: chunks the answer cites
    sql: str | None             # SQL_QUERY: generated SQL
    rows: list[dict] | None     # SQL_QUERY: result rows
    error: str | None
    answer: str


def classify_node(state: QueryState) -> dict[str, Any]:
    result = classify_query(state["question"])
    return {"route": result["route"], "route_reason": result["reason"]}


def document_rag_node(state: QueryState) -> dict[str, Any]:
    try:
        from ida.generation.answer import generate_rag_answer  # lazy
        from ida.retrieval.pipeline import retrieve

        chunks = retrieve(state["question"], top_k=5)
        result = generate_rag_answer(state["question"], chunks)
        return {"chunks": chunks, "answer": result["answer"], "citations": result["citations"]}
    except Exception as e:
        return {"answer": "Sorry, something went wrong while searching the handbook.", "error": str(e)}


def sql_query_node(state: QueryState) -> dict[str, Any]:
    try:
        from ida.generation.answer import explain_sql_result  # lazy
        from ida.retrieval.sql_tool import nl_to_sql_query

        result = nl_to_sql_query(state["question"])
        answer = explain_sql_result(
            state["question"], result["sql"], result["rows"], result["error"]
        )
        return {
            "sql": result["sql"],
            "rows": result["rows"],
            "error": result["error"],
            "answer": answer,
        }
    except Exception as e:
        return {"answer": "Sorry, something went wrong while querying the records.", "error": str(e)}


def direct_answer_node(state: QueryState) -> dict[str, Any]:
    try:
        from ida.generation.answer import generate_direct_answer  # lazy

        return {"answer": generate_direct_answer(state["question"])}
    except Exception as e:
        return {"answer": "Sorry, I couldn't generate a reply just now.", "error": str(e)}


def _pick_branch(
    state: QueryState,
) -> Literal["document_rag", "sql_query", "direct_answer"]:
    return {
        "DOCUMENT_RAG": "document_rag",
        "SQL_QUERY": "sql_query",
        "DIRECT_ANSWER": "direct_answer",
    }[state["route"]]


def build_graph():
    g = StateGraph(QueryState)
    g.add_node("classify", classify_node)
    g.add_node("document_rag", document_rag_node)
    g.add_node("sql_query", sql_query_node)
    g.add_node("direct_answer", direct_answer_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", _pick_branch)
    g.add_edge("document_rag", END)
    g.add_edge("sql_query", END)
    g.add_edge("direct_answer", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_query(question: str) -> dict:
    """
    The one entry point the API, the UI, and the eval harness all call.

    Returns: {question, route, route_reason, answer, citations, sql, rows,
              error, elapsed_s}
    """
    start = time.perf_counter()
    out = get_graph().invoke({"question": question})
    return {
        "question": question,
        "route": out.get("route"),
        "route_reason": out.get("route_reason"),
        "answer": out.get("answer", ""),
        "citations": out.get("citations", []),
        "sql": out.get("sql"),
        "rows": out.get("rows"),
        "error": out.get("error"),
        "elapsed_s": round(time.perf_counter() - start, 2),
    }


if __name__ == "__main__":
    for q in ["Hello!", "What is FERPA?", "How many students have unpaid fees?"]:
        r = run_query(q)
        print(f"\n[{r['route']}] {q}\n{r['answer']}")
