from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from src.astra_store import verify_astra_connection
from src.config import Settings
from src.documents import (
    ingest_pdf,
    load_pdf_documents,
    split_documents,
    stable_document_ids,
)
from src.providers import create_chat_model, create_embeddings
from src.rag import OUT_OF_SCOPE_ANSWER, answer_question


def _settings(**overrides) -> Settings:
    values = {
        "ASTRA_DB_API_ENDPOINT": "https://example.astra.datastax.com",
        "ASTRA_DB_APPLICATION_TOKEN": "AstraCS-test-value",
        "ASTRA_DB_NAMESPACE": "",
        "ASTRA_DB_COLLECTION_NAME": "pdf_rag",
        "EMBEDDING_PROVIDER": "huggingface",
        "LLM_PROVIDER": "huggingface",
        "PDF_CHUNK_SIZE": "80",
        "PDF_CHUNK_OVERLAP": "10",
        "RAG_TOP_K": "4",
        "RAG_MIN_RELEVANCE_SCORE": "0.35",
    }
    values.update(overrides)
    return Settings.from_env(values)


class FakeStore:
    def __init__(self, matches):
        self.matches = matches
        self.added_documents = []
        self.added_ids = []

    def similarity_search_with_relevance_scores(self, query, k):
        return self.matches[:k]

    def similarity_search(self, query, k):
        if not self.added_documents:
            return []
        return [self.added_documents[0]]

    def add_documents(self, documents, ids):
        self.added_documents = list(documents)
        self.added_ids = list(ids)


class FakeChatModel:
    def __init__(self):
        self.last_prompt = None
        self.last_prompt_text = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        messages = prompt.to_messages()
        self.last_prompt_text = str(messages[-1].content)
        context = self.last_prompt_text.split("PDF excerpts:\n", maxsplit=1)[1]
        context = context.split("\n\nQuestion:\n", maxsplit=1)[0]
        return f"Retrieved evidence:\n{context}"


class SettingsTests(unittest.TestCase):
    def test_settings_validate_required_credentials_and_chunking(self):
        with self.assertRaisesRegex(ValueError, "ASTRA_DB_API_ENDPOINT"):
            Settings.from_env({})
        with self.assertRaisesRegex(ValueError, "PDF_CHUNK_OVERLAP"):
            _settings(PDF_CHUNK_SIZE="100", PDF_CHUNK_OVERLAP="100")
        with self.assertRaisesRegex(ValueError, "valid HTTP"):
            _settings(ASTRA_DB_API_ENDPOINT="not-a-url")


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.pages = [
            Document(
                page_content=(
                    "The Northside Community Garden Pilot began in April 2025. "
                    "It serves 48 households and includes 12 raised beds. "
                )
                * 5,
                metadata={"source": "garden_report.pdf", "page": 0},
            ),
            Document(
                page_content=(
                    "The project uses 600 liters of water each week and composts "
                    "180 kilograms of material per month. "
                )
                * 4,
                metadata={"source": "garden_report.pdf", "page": 1},
            ),
        ]

    def test_splitter_creates_overlapping_chunks_with_page_metadata(self):
        chunks = split_documents(self.pages, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all("start_index" in chunk.metadata for chunk in chunks))
        self.assertEqual({chunk.metadata["page"] for chunk in chunks}, {0, 1})

    def test_sample_pdf_loads_all_pages_and_expected_content(self):
        sample_path = Path(__file__).resolve().parents[1] / "data" / "garden_report.pdf"
        pages = load_pdf_documents(sample_path)
        text = "\n".join(page.page_content for page in pages)

        self.assertEqual(len(pages), 3)
        self.assertIn("12 raised beds", text)
        self.assertIn("600 liters of water each week", text)

    def test_document_ids_are_stable_and_content_sensitive(self):
        first = stable_document_ids(self.pages)
        second = stable_document_ids(self.pages)
        changed = [Document("changed content", metadata=self.pages[0].metadata)]
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], stable_document_ids(changed)[0])

    def test_ingestion_stores_chunks_and_verifies_readback(self):
        settings = _settings()
        store = FakeStore([])
        with patch("src.documents.load_pdf_documents", return_value=self.pages):
            result = ingest_pdf("garden_report.pdf", store, settings)

        self.assertEqual(result.pages_loaded, 2)
        self.assertEqual(result.chunks_stored, len(store.added_documents))
        self.assertEqual(result.document_ids, tuple(store.added_ids))
        self.assertEqual(
            store.added_documents[0].metadata["ingest_id"],
            store.added_ids[0],
        )


class RagTests(unittest.TestCase):
    def test_answers_from_relevant_context_with_page_sources(self):
        document = Document(
            page_content="The pilot serves 48 households.",
            metadata={"source": "garden_report.pdf", "page": 1},
        )
        store = FakeStore([(document, 0.91)])
        model = FakeChatModel()
        response = answer_question(
            "How many households are served?",
            store,
            model,
            min_relevance_score=0.35,
        )

        self.assertIn("48 households", response.answer)
        self.assertEqual(response.sources[0].page, 2)
        self.assertIn("The pilot serves 48 households.", model.last_prompt_text)
        self.assertIn("How many households are served?", model.last_prompt_text)

    def test_out_of_context_question_is_refused_without_calling_model(self):
        store = FakeStore([(Document("Garden beds are raised.", metadata={}), 0.12)])
        model = FakeChatModel()
        response = answer_question(
            "What is the capital of France?",
            store,
            model,
            min_relevance_score=0.35,
        )

        self.assertEqual(response.answer, OUT_OF_SCOPE_ANSWER)
        self.assertEqual(response.sources, ())
        self.assertIsNone(model.last_prompt)

    def test_five_sample_questions_receive_matching_pdf_evidence(self):
        sample_path = Path(__file__).resolve().parents[1] / "data" / "garden_report.pdf"
        pages = load_pdf_documents(sample_path)
        chunks = split_documents(pages, chunk_size=300, chunk_overlap=40)
        cases = [
            ("When did the pilot begin?", "April 2025"),
            ("How many raised beds are there?", "12 raised beds"),
            ("How much water is used each week?", "600 liters of water each week"),
            ("How many households does it serve?", "48 Northside households"),
            ("How much compost is processed per month?", "180 kilograms of plant"),
        ]

        for question, evidence in cases:
            with self.subTest(question=question):
                matched_chunk = next(
                    chunk for chunk in chunks if evidence in chunk.page_content
                )
                model = FakeChatModel()
                response = answer_question(
                    question,
                    FakeStore([(matched_chunk, 0.95)]),
                    model,
                    min_relevance_score=0.35,
                )
                self.assertIn(evidence, response.answer)
                self.assertTrue(response.sources)

    def test_empty_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Question cannot be empty"):
            answer_question("  ", FakeStore([]), FakeChatModel())


class AstraConnectionTests(unittest.TestCase):
    def test_connection_verification_uses_endpoint_token_and_keyspace(self):
        calls = {}

        class FakeDatabase:
            def info(self):
                return {"status": "ok"}

        class FakeClient:
            def get_database(self, endpoint, **kwargs):
                calls["endpoint"] = endpoint
                calls.update(kwargs)
                return FakeDatabase()

        fake_astrapy = types.ModuleType("astrapy")
        fake_astrapy.DataAPIClient = FakeClient
        settings = _settings(ASTRA_DB_NAMESPACE="garden_keyspace")
        with patch.dict(sys.modules, {"astrapy": fake_astrapy}):
            result = verify_astra_connection(settings)

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(calls["endpoint"], settings.api_endpoint)
        self.assertEqual(calls["token"], settings.application_token)
        self.assertEqual(calls["keyspace"], "garden_keyspace")

    def test_vector_store_factory_passes_astra_settings(self):
        created = {}

        class FakeVectorStore:
            def __init__(self, **kwargs):
                created.update(kwargs)

        fake_package = types.ModuleType("langchain_astradb")
        fake_package.AstraDBVectorStore = FakeVectorStore
        settings = _settings()
        embeddings = object()
        with patch.dict(sys.modules, {"langchain_astradb": fake_package}):
            from src.astra_store import create_vector_store

            store = create_vector_store(settings, embeddings)

        self.assertIsInstance(store, FakeVectorStore)
        self.assertIs(created["embedding"], embeddings)
        self.assertEqual(created["collection_name"], settings.collection_name)
        self.assertEqual(created["api_endpoint"], settings.api_endpoint)
        self.assertEqual(created["token"], settings.application_token)


class ProviderTests(unittest.TestCase):
    def test_huggingface_embedding_and_chat_model_factories(self):
        calls = {}

        class FakeEndpoint:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                calls["endpoint"] = kwargs

        class FakeEmbeddings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                calls["embeddings"] = kwargs

        class FakeChat:
            def __init__(self, llm):
                self.llm = llm

        fake_package = types.ModuleType("langchain_huggingface")
        fake_package.HuggingFaceEndpoint = FakeEndpoint
        fake_package.HuggingFaceEndpointEmbeddings = FakeEmbeddings
        fake_package.ChatHuggingFace = FakeChat
        environment = {
            "HUGGINGFACEHUB_API_TOKEN": "test-token",
            "HF_EMBEDDING_MODEL": "example/embeddings",
            "HF_LLM_MODEL": "example/chat",
            "HF_PROVIDER": "auto",
        }
        with patch.dict(sys.modules, {"langchain_huggingface": fake_package}):
            with patch.dict(os.environ, environment, clear=True):
                embeddings = create_embeddings(_settings())
                chat_model = create_chat_model(_settings())

        self.assertIsInstance(embeddings, FakeEmbeddings)
        self.assertEqual(calls["embeddings"]["model"], "example/embeddings")
        self.assertEqual(calls["embeddings"]["task"], "feature-extraction")
        self.assertEqual(calls["embeddings"]["huggingfacehub_api_token"], "test-token")
        self.assertIsInstance(chat_model, FakeChat)
        self.assertIsInstance(chat_model.llm, FakeEndpoint)
        self.assertEqual(calls["endpoint"]["repo_id"], "example/chat")
        self.assertEqual(calls["endpoint"]["task"], "text-generation")

    def test_openai_embedding_requires_api_key(self):
        settings = _settings(EMBEDDING_PROVIDER="openai")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                create_embeddings(settings)


if __name__ == "__main__":
    unittest.main()
