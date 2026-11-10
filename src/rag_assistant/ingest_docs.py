"""
Document ingestion for the RAG assistant.

Chunks operations documents (SOPs, manuals, reports) and stores
embeddings in ChromaDB for retrieval.

Markdown files are split on headers first, so every chunk knows the
section it came from (used for citations), then into ~512-character
chunks with 50 characters of overlap.
"""

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.utils.config import CHROMA_PERSIST_DIR, DOCS_DIR

# Embedding model — runs locally, no API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "ops_docs"

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the embedding model (loaded once per process)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(docs_dir: Path | None = None) -> list[Document]:
    """
    Load all documents from the docs directory.

    Supports .md and .txt files, plus .pdf when the optional `pypdf`
    package is installed. `source` metadata is the file name.
    """
    docs_dir = Path(docs_dir) if docs_dir is not None else DOCS_DIR
    documents = []

    for path in sorted(docs_dir.glob("*.md")) + sorted(docs_dir.glob("*.txt")):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name, "format": path.suffix.lstrip(".")},
            )
        )

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if pdf_files:
        try:
            from langchain_community.document_loaders import PyPDFLoader

            for path in pdf_files:
                for page in PyPDFLoader(str(path)).load():
                    page.metadata = {
                        "source": path.name,
                        "format": "pdf",
                        "section": f"page {page.metadata.get('page', 0) + 1}",
                    }
                    documents.append(page)
        except ImportError:
            print(f"Skipping {len(pdf_files)} PDF(s): pip install pypdf to ingest PDFs")

    print(f"Loaded {len(documents)} document(s) from {docs_dir}")
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Split documents into chunks for embedding.

    Markdown is split by section first; each chunk gets a `section`
    metadata field like "Shipping & Logistics Policy > 3. Late Delivery Tracking".
    """
    header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    sections = []
    for doc in documents:
        if doc.metadata.get("format") == "md":
            for part in header_splitter.split_text(doc.page_content):
                titles = [
                    part.metadata[key]
                    for (key,) in [("h1",), ("h2",), ("h3",)]
                    if key in part.metadata
                ]
                sections.append(
                    Document(
                        page_content=part.page_content,
                        metadata={**doc.metadata, "section": " > ".join(titles)},
                    )
                )
        else:
            sections.append(doc)

    chunks = text_splitter.split_documents(sections)
    print(f"Split into {len(chunks)} chunks (size={chunk_size} chars, overlap={chunk_overlap})")
    return chunks


def _open_store(persist_dir: str) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_vector_store(chunks: list[Document], persist_dir: str | None = None) -> Chroma:
    """Replace the vector store's contents with `chunks` (re-running never duplicates)."""
    persist_dir = persist_dir or CHROMA_PERSIST_DIR

    store = _open_store(persist_dir)
    store.delete_collection()
    store = _open_store(persist_dir)
    if chunks:
        ids = [f"{c.metadata['source']}#{i}" for i, c in enumerate(chunks)]
        store.add_documents(chunks, ids=ids)
    get_vector_store.cache_clear()

    print(f"Created vector store with {len(chunks)} chunks at {persist_dir}")
    return store


@lru_cache(maxsize=4)
def get_vector_store(persist_dir: str | None = None) -> Chroma:
    """Open the existing vector store (cached per process)."""
    return _open_store(persist_dir or CHROMA_PERSIST_DIR)


def ingest_all(docs_dir: Path | None = None, persist_dir: str | None = None) -> Chroma:
    """Full ingestion pipeline: load -> chunk -> embed -> store."""
    documents = load_documents(docs_dir)
    if not documents:
        print("No documents found. Add .md, .txt, or .pdf files to data/docs/")
    return build_vector_store(chunk_documents(documents), persist_dir)


if __name__ == "__main__":
    ingest_all()
