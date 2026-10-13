"""
T20: Streamlit UI -- question in, routed + cited answer out.

Needs the API running:  uv run uvicorn ida.api.main:app --port 8000
Run the UI:             uv run streamlit run ui/streamlit_app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("IDA_API_URL", "http://localhost:8000")

ROUTE_LABELS = {
    "DOCUMENT_RAG": "📄 Handbook search",
    "SQL_QUERY": "🗄️ Records database",
    "DIRECT_ANSWER": "💬 Direct answer",
}

EXAMPLES = {
    "Handbook": [
        "What is the alcohol policy on campus?",
        "How do I request a leave of absence?",
        "What happens if I get caught cheating on an exam?",
    ],
    "Records": [
        "How many students have unpaid fees?",
        "What is the average grade points per program?",
        "Show the top 5 students by number of disciplinary incidents.",
    ],
    "Chat": ["What can you help me with?"],
}

st.set_page_config(page_title="Intelligent Document Assistant", page_icon="🎓")
st.title("🎓 Intelligent Document Assistant")
st.caption("Ask about the student handbook or the student records. The assistant decides where to look.")

if "question" not in st.session_state:
    st.session_state.question = ""

with st.sidebar:
    st.header("Try an example")
    for group, questions in EXAMPLES.items():
        st.subheader(group)
        for q in questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.question = q
    st.divider()
    st.caption(f"API: {API_URL}")

question = st.text_input("Your question", key="question")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(f"{API_URL}/query", json={"question": question.strip()}, timeout=120)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Can't reach the API at {API_URL}. Is uvicorn running?")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    route = result.get("route")
    st.caption(
        f"**Routed to:** {ROUTE_LABELS.get(route, route)}  ·  "
        f"{result.get('route_reason', '')}  ·  {result.get('elapsed_s', '?')}s"
    )

    st.markdown(result.get("answer") or "_No answer returned._")

    if result.get("error") and route != "SQL_QUERY":
        st.warning(f"Something went wrong behind the scenes: {result['error']}")

    citations = result.get("citations") or []
    if citations:
        st.subheader("Sources")
        for c in citations:
            pages = (
                f"p.{c['page_start']}"
                if c["page_start"] == c["page_end"]
                else f"p.{c['page_start']}-{c['page_end']}"
            )
            with st.expander(f"[{c['n']}] {c['section']} ({pages})"):
                st.write(c["text"])

    if result.get("sql"):
        with st.expander("SQL used"):
            st.code(result["sql"], language="sql")
        rows = result.get("rows")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
