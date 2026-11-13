from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from src.astra_store import create_vector_store
from src.config import Settings
from src.documents import ingest_pdf
from src.providers import create_chat_model, create_embeddings
from src.rag import answer_question

load_dotenv()
st.set_page_config(page_title="AstraDB PDF RAG", page_icon="📄", layout="wide")

st.title("AstraDB PDF Question Answering")
st.caption(
    "Upload a PDF to index it in AstraDB, then ask questions grounded in its contents."
)

with st.sidebar:
    st.subheader("Configuration")
    st.write(f"Collection: `{os.getenv('ASTRA_DB_COLLECTION_NAME', 'pdf_rag')}`")
    st.caption("Credentials are read from your local .env file and are never displayed.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["rag_messages"] = []
        st.session_state["rag_session_id"] = uuid4().hex
        st.rerun()

if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []
if "rag_session_id" not in st.session_state:
    st.session_state["rag_session_id"] = uuid4().hex
if "rag_indexed_files" not in st.session_state:
    st.session_state["rag_indexed_files"] = {}
if "rag_components" not in st.session_state:
    st.session_state["rag_components"] = None


def _get_components():
    if st.session_state["rag_components"] is None:
        settings = Settings.from_env()
        embeddings = create_embeddings(settings)
        st.session_state["rag_components"] = (
            settings,
            create_vector_store(settings, embeddings),
            create_chat_model(settings),
        )
    return st.session_state["rag_components"]


uploaded_pdf = st.file_uploader("Choose a PDF document", type=["pdf"])
if uploaded_pdf and st.button("Ingest PDF", type="primary"):
    file_bytes = uploaded_pdf.getvalue()
    fingerprint = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state["rag_indexed_files"].get(uploaded_pdf.name) == fingerprint:
        st.info("This exact PDF is already indexed in the current collection.")
    else:
        try:
            settings, vector_store, _ = _get_components()
        except ValueError as exc:
            st.error(str(exc))
        else:
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".pdf",
                    prefix="astra-rag-",
                    delete=False,
                ) as temporary_file:
                    temporary_file.write(file_bytes)
                    temporary_path = Path(temporary_file.name)
                with st.spinner("Reading, splitting, embedding, and storing the PDF..."):
                    result = ingest_pdf(temporary_path, vector_store, settings)
                st.session_state["rag_indexed_files"][uploaded_pdf.name] = fingerprint
                st.success(
                    f"Indexed {result.pages_loaded} pages into "
                    f"{result.chunks_stored} persistent AstraDB chunks."
                )
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

for turn in st.session_state["rag_messages"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Retrieved PDF sources"):
                for source in turn["sources"]:
                    page_label = f", page {source['page']}" if source["page"] else ""
                    st.markdown(
                        f"- `{source['source']}`{page_label} "
                        f"(relevance {source['relevance_score']:.2f})"
                    )

question = st.chat_input("Ask a question about the indexed PDF")
if question:
    st.session_state["rag_messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            settings, vector_store, chat_model = _get_components()
        except ValueError as exc:
            st.error(str(exc))
        else:
            with st.spinner("Searching the PDF and preparing a grounded answer..."):
                response = answer_question(
                    question,
                    vector_store,
                    chat_model,
                    top_k=settings.top_k,
                    min_relevance_score=settings.min_relevance_score,
                )
            st.markdown(response.answer)
            source_rows = [
                {
                    "source": source.source,
                    "page": source.page,
                    "relevance_score": source.relevance_score,
                }
                for source in response.sources
            ]
            if source_rows:
                with st.expander("Retrieved PDF sources", expanded=True):
                    for source in source_rows:
                        page_label = f", page {source['page']}" if source["page"] else ""
                        st.markdown(
                            f"- `{source['source']}`{page_label} "
                            f"(relevance {source['relevance_score']:.2f})"
                        )
            st.session_state["rag_messages"].append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": source_rows,
                }
            )
