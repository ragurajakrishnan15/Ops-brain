"""Operations Assistant — chat over the data (SQL) and documents (RAG)."""

import streamlit as st

from src.utils.config import GOOGLE_API_KEY

EXAMPLES = [
    "Which 5 categories had the most late deliveries in 2017?",
    "What is the average shipping delay by shipping mode?",
    "Do canceled orders count as demand?",
    "How is safety stock calculated?",
]

st.header("Operations Assistant")
st.caption(
    "Ask about supply chain data or operations documents. Data questions are answered with a "
    "read-only SQL query; document questions with retrieval over the SOPs (cited by section)."
)

if not GOOGLE_API_KEY:
    st.info(
        "The assistant needs a Google Gemini API key.\n\n"
        "1. Create a free key at https://aistudio.google.com/apikey\n"
        "2. Add `GOOGLE_API_KEY=...` to the `.env` file in the project root\n"
        "3. Restart the dashboard"
    )
    st.stop()


def render(message: dict) -> None:
    """Render one assistant reply with its SQL/results or document sources."""
    if message.get("mode"):
        st.caption("📊 Data query" if message["mode"] == "data" else "📄 Document search")
    if message.get("error"):
        st.error(message["content"])
    else:
        st.markdown(message["content"])

    if message.get("mode") == "data":
        results = message.get("results")
        if results is not None and len(results):
            note = " (first 200 rows)" if message.get("truncated") else ""
            with st.expander(f"Results — {len(results)} rows{note}"):
                st.dataframe(results, use_container_width=True, hide_index=True)
        if message.get("sql"):
            with st.expander("SQL"):
                st.code(message["sql"], language="sql")

    if message.get("mode") == "docs" and message.get("sources"):
        with st.expander("Sources"):
            for src in message["sources"]:
                section = f" — {src['section']}" if src.get("section") else ""
                st.write(f"- `{src['source']}`{section} (relevance {src['relevance']:.2f})")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render(message)

prompt = st.chat_input("Ask about operations data or documents...")
if not st.session_state.messages and not prompt:
    st.write("Try one of these:")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            prompt = example

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from src.rag_assistant.chat import chat

                result = chat(prompt)
                reply = {
                    "role": "assistant",
                    "content": result["answer"],
                    "mode": result["mode"],
                    "sql": result.get("sql"),
                    "results": result.get("results"),
                    "truncated": result.get("truncated", False),
                    "sources": result.get("sources"),
                    "error": bool(result.get("error")),
                }
            except Exception as e:
                reply = {
                    "role": "assistant",
                    "content": f"Error: {e}. Make sure the data pipeline and document ingestion have run.",
                    "error": True,
                }
        render(reply)

    st.session_state.messages.append(reply)
