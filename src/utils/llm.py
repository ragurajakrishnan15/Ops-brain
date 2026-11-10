"""
LLM access — one place to configure the chat model used by the assistant.

Uses Google Gemini through LangChain. Model names come from config so they
can be changed in .env without touching code.

Check your key and model names with:
    python -m src.utils.llm
"""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.config import GEMINI_MODEL, GEMINI_ROUTER_MODEL, GOOGLE_API_KEY


@lru_cache(maxsize=None)
def get_llm(
    role: str = "answer",
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> ChatGoogleGenerativeAI:
    """
    Return a cached Gemini chat model.

    role="router" uses the cheaper, faster router model; anything else uses
    the main model. Output limits are generous because Gemini's thinking
    tokens count toward max_output_tokens.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Create a key at https://aistudio.google.com/apikey "
            "and add it to .env"
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_ROUTER_MODEL if role == "router" else GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def response_text(response) -> str:
    """
    Extract plain text from a LangChain chat response.

    Newer Gemini models can return content as a list of parts
    (text plus thinking/signature blocks) instead of a single string.
    """
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        part if isinstance(part, str) else part.get("text", "")
        for part in content
        if isinstance(part, str) or part.get("type", "text") == "text"
    )


if __name__ == "__main__":
    for role in ("router", "answer"):
        llm = get_llm(role)
        reply = response_text(llm.invoke("Reply with exactly: OK"))
        print(f"{role:>6} model {llm.model}: {reply.strip()!r}")
