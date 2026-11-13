from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_endpoint: str
    application_token: str = field(repr=False)
    namespace: str | None
    collection_name: str
    embedding_provider: str
    llm_provider: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_relevance_score: float

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        if environ is None:
            load_dotenv()
            source = os.environ
        else:
            source = environ

        endpoint = source.get("ASTRA_DB_API_ENDPOINT", "").strip()
        token = source.get("ASTRA_DB_APPLICATION_TOKEN", "").strip()
        if not endpoint:
            raise ValueError("ASTRA_DB_API_ENDPOINT must be configured.")
        if not token:
            raise ValueError("ASTRA_DB_APPLICATION_TOKEN must be configured.")

        parsed_endpoint = urlparse(endpoint)
        if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
            raise ValueError("ASTRA_DB_API_ENDPOINT must be a valid HTTP(S) URL.")

        collection_name = source.get("ASTRA_DB_COLLECTION_NAME", "pdf_rag").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", collection_name):
            raise ValueError(
                "ASTRA_DB_COLLECTION_NAME must contain 1-48 letters, digits, "
                "underscores, or hyphens."
            )

        embedding_provider = source.get("EMBEDDING_PROVIDER", "huggingface").casefold()
        llm_provider = source.get("LLM_PROVIDER", "huggingface").casefold()
        if embedding_provider not in {"huggingface", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER must be huggingface or openai.")
        if llm_provider not in {"huggingface", "openai"}:
            raise ValueError("LLM_PROVIDER must be huggingface or openai.")

        chunk_size = _integer_setting(source, "PDF_CHUNK_SIZE", 1000)
        chunk_overlap = _integer_setting(source, "PDF_CHUNK_OVERLAP", 150)
        top_k = _integer_setting(source, "RAG_TOP_K", 4)
        score_text = source.get("RAG_MIN_RELEVANCE_SCORE", "0.35")
        try:
            min_relevance_score = float(score_text)
        except ValueError as exc:
            raise ValueError("RAG_MIN_RELEVANCE_SCORE must be a number from 0 to 1.") from exc

        if chunk_size < 1:
            raise ValueError("PDF_CHUNK_SIZE must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "PDF_CHUNK_OVERLAP must be non-negative and smaller than PDF_CHUNK_SIZE."
            )
        if top_k < 1:
            raise ValueError("RAG_TOP_K must be greater than zero.")
        if not 0 <= min_relevance_score <= 1:
            raise ValueError("RAG_MIN_RELEVANCE_SCORE must be between 0 and 1.")

        return cls(
            api_endpoint=endpoint.rstrip("/"),
            application_token=token,
            namespace=source.get("ASTRA_DB_NAMESPACE", "").strip() or None,
            collection_name=collection_name,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            min_relevance_score=min_relevance_score,
        )


def _integer_setting(source: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(source.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
