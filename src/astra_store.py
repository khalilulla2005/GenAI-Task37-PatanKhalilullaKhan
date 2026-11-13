from __future__ import annotations

from src.config import Settings


def verify_astra_connection(settings: Settings):
    from astrapy import DataAPIClient

    client = DataAPIClient()
    database_options = {"token": settings.application_token}
    if settings.namespace:
        database_options["keyspace"] = settings.namespace
    database = client.get_database(settings.api_endpoint, **database_options)
    return database.info()


def create_vector_store(settings: Settings, embeddings):
    from langchain_astradb import AstraDBVectorStore

    return AstraDBVectorStore(
        collection_name=settings.collection_name,
        embedding=embeddings,
        api_endpoint=settings.api_endpoint,
        token=settings.application_token,
        namespace=settings.namespace,
    )
