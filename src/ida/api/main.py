"""
T19: FastAPI app.

    POST /query      router -> path -> generation (the main endpoint)
    POST /sql_query  NL -> SQL directly, bypassing the router (T13 endpoint)
    GET  /health

Run: uv run uvicorn ida.api.main:app --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ida.router.graph import run_query


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model and re-ranker now, so the first real request
    # isn't the slow one. A failed warm-up shouldn't stop the server starting.
    try:
        from ida.retrieval.pipeline import retrieve

        retrieve("warm up", top_k=1)
    except Exception as e:
        print(f"[warmup] skipped: {e}")
    yield


app = FastAPI(title="Intelligent Document Assistant", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest):
    return run_query(req.question.strip())


@app.post("/sql_query")
def sql_query(req: QueryRequest):
    from ida.retrieval.sql_tool import nl_to_sql_query

    return nl_to_sql_query(req.question.strip())
