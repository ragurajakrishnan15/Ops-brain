"""
Unified chat interface — routes queries to either RAG (documents) or SQL (data).

The router decides based on the question: if it's about processes, policies,
or SOPs, it goes to RAG. If it's about numbers, metrics, or data, it goes
to the SQL agent.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from src.rag_assistant.retriever import retrieve_with_scores
from src.rag_assistant.sql_agent import ask_data
from src.utils.llm import get_llm, response_text

ROUTER_PROMPT = """Classify this question into one of two categories:
- "data" if the question asks about numbers, metrics, trends, counts, comparisons, or any quantitative analysis that would need a database query
- "docs" if the question asks about processes, procedures, policies, definitions, how things work, or any qualitative information that would be in documentation

Respond with ONLY the word "data" or "docs".

Question: {question}"""

RAG_SYSTEM_PROMPT = """You are an operations analytics assistant. Answer the user's
question using ONLY the provided context from operations documents.

Rules:
- If the context doesn't contain the answer, say "I don't have that information in the available documents."
- Always cite your sources by naming the document (and section) the information came from.
- Be concise and direct.
- If the question is about data or metrics, suggest they ask the data assistant instead.

Context from documents:
{context}"""


def route_query(question: str) -> str:
    """Classify a question as 'data' or 'docs'."""
    llm = get_llm("router", temperature=0.0)

    response = llm.invoke(ROUTER_PROMPT.format(question=question))
    classification = response_text(response).strip().lower()

    if "data" in classification:
        return "data"
    return "docs"


def ask_docs(question: str) -> dict:
    """Answer a question using RAG over operations documents."""
    # Retrieve relevant chunks
    chunks = retrieve_with_scores(question, top_k=5)

    if not chunks:
        return {
            "answer": "No relevant documents found. Please add operations documents to the data/docs/ directory.",
            "sources": [],
            "mode": "docs",
        }

    # Build context from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Section: {c['section']}]\n{c['content']}" for c in chunks
    )

    # Generate answer with Gemini
    llm = get_llm("answer", temperature=0.1)

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)

    return {
        "answer": response_text(response),
        "sources": [
            {"source": c["source"], "section": c["section"], "relevance": c["score"]}
            for c in chunks
        ],
        "mode": "docs",
    }


def chat(question: str) -> dict:
    """
    Main entry point: route the question and return an answer.

    Returns a dict with 'answer', 'mode' ('data' or 'docs'),
    and mode-specific metadata (SQL query or document sources).
    """
    mode = route_query(question)

    if mode == "data":
        result = ask_data(question)
        if result["error"]:
            answer = (
                "I couldn't answer that from the data — the query failed twice.\n\n"
                f"Last error: {result['error']}"
            )
        elif result["row_count"] == 0:
            answer = "The query ran but returned no rows."
        else:
            answer = result["summary"]

        return {
            "answer": answer,
            "mode": "data",
            "sql": result["sql"],
            "results": result["results"],
            "truncated": result["truncated"],
            "row_count": result["row_count"],
            "error": result["error"],
        }
    else:
        return ask_docs(question)
