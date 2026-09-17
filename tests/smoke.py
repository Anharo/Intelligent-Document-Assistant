import os, psycopg
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient

load_dotenv()
print("Qdrant:", QdrantClient(url=os.getenv("QDRANT_URL")).get_collections())
with psycopg.connect(os.getenv("POSTGRES_DSN")) as c:
    print("Postgres:", c.execute("select version()").fetchone()[0][:20])
r = Groq(api_key=os.getenv("GROQ_API_KEY")).chat.completions.create(
    model=os.getenv("GROQ_MODEL"),
    messages=[{"role": "user", "content": "reply with OK"}],
    max_tokens=50,
)
print("Groq:", r.choices[0].message.content)