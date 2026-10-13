"""
Thin Groq chat wrapper shared by every generation step.

One client, one place for retry-on-rate-limit, so free-tier 429s don't turn
into crashes halfway through a demo or an evaluation run.
"""
import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Optional: "low" | "medium" | "high". Only valid for reasoning models such as
# gpt-oss; leave unset for Llama. "low" makes routing/explanations much faster.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT")

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def chat(system: str, user: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
    """
    Returns the model's reply text ('' if the model returned nothing).
    Retries up to 3 times, but only on rate-limit style errors; anything else
    (bad key, bad model name) raises immediately.
    """
    extra = {"reasoning_effort": GROQ_REASONING_EFFORT} if GROQ_REASONING_EFFORT else {}
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = _get_client().chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            if "429" in msg or "rate" in msg:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise last_error  # type: ignore[misc]
