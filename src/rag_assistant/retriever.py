"""
Document retriever — finds the most relevant chunks for a user query.
"""

from src.rag_assistant.ingest_docs import get_vector_store


def get_retriever(top_k: int = 5, persist_dir: str | None = None):
    """
    Return a LangChain retriever over the operations documents.

    Args:
        top_k: Number of chunks to retrieve per query.
    """
    return get_vector_store(persist_dir).as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def retrieve_with_scores(query: str, top_k: int = 5, persist_dir: str | None = None) -> list[dict]:
    """
    Retrieve documents with cosine-similarity relevance scores (higher = closer).

    Returns a list of dicts with 'content', 'source', 'section' and 'score'.
    Useful for displaying citations in the chat interface.
    """
    results = get_vector_store(persist_dir).similarity_search_with_relevance_scores(query, k=top_k)

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "section": doc.metadata.get("section", ""),
            "score": round(score, 3),
        }
        for doc, score in results
    ]
