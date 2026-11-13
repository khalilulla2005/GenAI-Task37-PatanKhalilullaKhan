from __future__ import annotations

import os

from src.config import Settings


def _huggingface_token() -> str:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) to use Hugging Face models."
        )
    return token


def create_embeddings(settings: Settings):
    if settings.embedding_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=api_key,
        )

    token = _huggingface_token()
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    return HuggingFaceEndpointEmbeddings(
        model=os.getenv(
            "HF_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        task="feature-extraction",
        provider=os.getenv("HF_PROVIDER", "auto"),
        huggingfacehub_api_token=token,
    )


def create_chat_model(settings: Settings):
    if settings.llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI chat model.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=api_key,
        )

    token = _huggingface_token()
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id=os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        provider=os.getenv("HF_PROVIDER", "auto"),
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=endpoint)
