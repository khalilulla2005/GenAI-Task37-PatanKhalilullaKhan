from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.astra_store import create_vector_store, verify_astra_connection
from src.config import Settings
from src.documents import ingest_pdf
from src.providers import create_chat_model, create_embeddings
from src.rag import answer_question


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into AstraDB and ask grounded RAG questions."
    )
    parser.add_argument("--check-connection", action="store_true")
    parser.add_argument("--ingest-pdf", type=Path, help="Load, split, and store a PDF.")
    parser.add_argument("--question", help="Ask a question about previously indexed PDFs.")
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override RAG_TOP_K for one question.",
    )
    args = parser.parse_args()

    if args.check_connection and (args.ingest_pdf or args.question):
        parser.error("--check-connection cannot be combined with ingest or question options.")
    if not (args.check_connection or args.ingest_pdf or args.question):
        parser.print_help()
        return

    settings = Settings.from_env()
    if args.check_connection:
        verify_astra_connection(settings)
        print(
            "AstraDB connection verified for "
            f"{settings.api_endpoint} (collection: {settings.collection_name})."
        )
        return

    embeddings = create_embeddings(settings)
    vector_store = create_vector_store(settings, embeddings)
    if args.ingest_pdf:
        result = ingest_pdf(args.ingest_pdf, vector_store, settings)
        print(
            f"Loaded {result.pages_loaded} PDF pages and stored "
            f"{result.chunks_stored} chunks in '{settings.collection_name}'."
        )
        print("Persistence verified by retrieving the first ingested chunk.")

    if args.question:
        response = answer_question(
            args.question,
            vector_store,
            create_chat_model(settings),
            top_k=args.top_k if args.top_k is not None else settings.top_k,
            min_relevance_score=settings.min_relevance_score,
        )
        print(f"\nAnswer:\n{response.answer}")
        if response.sources:
            print("\nSources:")
            for source in response.sources:
                page_label = f", page {source.page}" if source.page is not None else ""
                print(
                    f"- {source.source}{page_label} "
                    f"(relevance {source.relevance_score:.2f})"
                )


if __name__ == "__main__":
    main()
