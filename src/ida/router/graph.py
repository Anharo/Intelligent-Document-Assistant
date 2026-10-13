"""
T16 (part 2): LangGraph state graph.

    START -> classify -> (conditional) -> document_rag | sql_query | direct_answer -> END

Week 4 scope: classification + the branch nodes gather their raw material
(retrieved chunks / SQL rows). Generation (turning that material into a final
natural-language answer) is Week 5 (T17/T18) and plugs into these same nodes.

Heavy imports (embedding model, reranker, Postgres) happen lazily INSIDE the
nodes, so routing on its own stays fast to import and test.
"""
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from ida.router.classifier import classify_query


class QueryState(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    chunks: list[dict]          # DOCUMENT_RAG: reranked chunks
    sql: str | None             # SQL_QUERY: generated SQL
    rows: list[dict] | None     # SQL_QUERY: result rows
    error: str | None
    answer: str                 # filled in by Week 5 generation nodes


def classify_node(state: QueryState) -> dict[str, Any]:
    result = classify_query(state["question"])
    return {"route": result["route"], "route_reason": result["reason"]}


def document_rag_node(state: QueryState) -> dict[str, Any]:
    from ida.retrieval.pipeline import retrieve  # lazy: loads models

    return {"chunks": retrieve(state["question"], top_k=5)}


def sql_query_node(state: QueryState) -> dict[str, Any]:
    from ida.retrieval.sql_tool import nl_to_sql_query  # lazy: needs Groq + Postgres

    result = nl_to_sql_query(state["question"])
    return {"sql": result["sql"], "rows": result["rows"], "error": result["error"]}


def direct_answer_node(state: QueryState) -> dict[str, Any]:
    # Week 5: call the LLM directly with no retrieved context.
    return {}


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


if __name__ == "__main__":
    graph = build_graph()
    for q in ["Hello!", "What is FERPA?", "How many students have unpaid fees?"]:
        out = graph.invoke({"question": q})
        print(f"{out['route']:<14} {q}")
