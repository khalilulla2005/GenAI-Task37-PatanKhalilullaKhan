from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings
from src.pdf_loader import PdfDocumentLoader


@dataclass(frozen=True)
class IngestionSummary:
    pages_loaded: int
    chunks_stored: int
    document_ids: tuple[str, ...]


def load_pdf_documents(pdf_path: str | Path) -> list[Document]:
    path = Path(pdf_path)
    if path.suffix.casefold() != ".pdf":
        raise ValueError("The uploaded document must be a PDF.")
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    documents = PdfDocumentLoader(path).load()
    if not documents:
        raise ValueError(f"No readable pages were found in {path.name}.")
    for document in documents:
        document.metadata["source"] = path.name
    return documents


def split_documents(
    documents: Sequence[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
    if not documents:
        raise ValueError("At least one document page is required.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(documents))
    if not chunks:
        raise ValueError("The PDF contained no text to index.")
    return chunks


def stable_document_ids(documents: Sequence[Document]) -> list[str]:
    ids = []
    for document in documents:
        identity = "\0".join(
            (
                str(document.metadata.get("source", "")),
                str(document.metadata.get("page", "")),
                str(document.metadata.get("start_index", "")),
                document.page_content,
            )
        )
        ids.append(hashlib.sha256(identity.encode("utf-8")).hexdigest())
    return ids


def ingest_pdf(pdf_path: str | Path, vector_store, settings: Settings) -> IngestionSummary:
    pages = load_pdf_documents(pdf_path)
    chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)
    ids = stable_document_ids(chunks)
    for chunk, document_id in zip(chunks, ids):
        chunk.metadata["ingest_id"] = document_id
    vector_store.add_documents(chunks, ids=ids)

    first_chunk_matches = vector_store.similarity_search(chunks[0].page_content, k=1)
    if not any(
        match.metadata.get("ingest_id") == ids[0]
        for match in first_chunk_matches
    ):
        raise RuntimeError(
            "AstraDB accepted the PDF chunks but the verification search could not "
            "retrieve the first chunk. Check the collection and embedding configuration."
        )

    return IngestionSummary(
        pages_loaded=len(pages),
        chunks_stored=len(chunks),
        document_ids=tuple(ids),
    )
