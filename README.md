# Assignment 37 — PDF RAG with AstraDB

A LangChain retrieval-augmented generation app that loads a PDF, splits it into attributed chunks, embeds and persists those chunks in DataStax AstraDB, then answers questions using retrieved evidence. It includes a Streamlit chat UI, a CLI, source-page reporting, an out-of-scope refusal, and offline tests.

## Requirements covered

1. **AstraDB setup and connection:** configure an Astra DB Serverless database endpoint, application token, and optional keyspace; the CLI has a connection-check command.
2. **LangChain and AstraDB:** `AstraDBVectorStore` persists and retrieves document chunks.
3. **PDF loading and splitting:** a small `langchain_core.document_loaders.BaseLoader` implementation powered by `pypdf`, plus `RecursiveCharacterTextSplitter`, preserves source page metadata and overlapping chunks without relying on the sunset `langchain-community` package.
4. **Embeddings and persistence:** choose Hugging Face or OpenAI embeddings; ingest is upserted with deterministic IDs and verified with a retrieval readback.
5. **Grounded PDF RAG:** retrieve relevant chunks, build a context-only prompt, generate an answer, and show PDF page sources.
6. **Validation and session state:** offline tests cover retrieval and refusal behavior; Streamlit stores the conversation in `st.session_state`, while AstraDB persists indexed chunks across app sessions.

## 1. Create and configure AstraDB

In the DataStax Astra portal:

1. Create an Astra DB Serverless vector database and wait until it is active.
2. Copy its **API endpoint** and generate a database application token with permission to read and write collections.
3. Optionally select or create the keyspace (namespace) you want to use.

## 2. Install and configure

Use Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set the Astra values in `.env`:

```text
ASTRA_DB_API_ENDPOINT=https://YOUR-DATABASE-REGION.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:YOUR_DATABASE_TOKEN
ASTRA_DB_NAMESPACE=default_keyspace
ASTRA_DB_COLLECTION_NAME=pdf_rag
```

By default, both embeddings and the chat model use Hugging Face Inference Providers. Configure `HUGGINGFACEHUB_API_TOKEN` in `.env`. To use OpenAI, set `EMBEDDING_PROVIDER=openai` and/or `LLM_PROVIDER=openai`, and configure `OPENAI_API_KEY`.

The first ingestion creates the configured AstraDB collection if it does not exist. AstraDB collections have a fixed vector dimension; changing embedding models or dimensions requires a new collection (or clearing/recreating the existing one). Never commit `.env`; it is excluded by Git.

## 3. Verify AstraDB connection

```powershell
python main.py --check-connection
```

This calls the AstraDB Data API using the endpoint and application token without printing the token.

## 4. Ingest a PDF

```powershell
python main.py --ingest-pdf "data\garden_report.pdf"
```

The pipeline loads pages, splits them (default 1,000 characters with 150-character overlap), assigns stable content-based IDs, embeds the chunks, writes them to AstraDB, and searches for the first chunk to verify readback. Re-ingesting the same unchanged PDF uses the same IDs to avoid creating duplicate chunks.

Chunk size and overlap are configurable through `.env`:

```text
PDF_CHUNK_SIZE=1000
PDF_CHUNK_OVERLAP=150
```

## 5. Ask a question

```powershell
python main.py --question "How many households does the garden serve?"
```

Or ingest and ask in one run:

```powershell
python main.py --ingest-pdf "data\garden_report.pdf" --question "How much water is used each week?"
```

The RAG flow retrieves the top relevant chunks, refuses to answer when every relevance score is below `RAG_MIN_RELEVANCE_SCORE`, and asks the selected language model to answer only from the excerpts. Sources include PDF filename and page. Tune `RAG_TOP_K` and `RAG_MIN_RELEVANCE_SCORE` for your model and document set.

## 6. Streamlit application

```powershell
streamlit run app.py
```

Upload a PDF, choose **Ingest PDF**, then ask questions in the chat. The conversation remains in `st.session_state` for the current browser session. Indexed chunks live in AstraDB and remain available after the app restarts.

## Five sample questions for `data/garden_report.pdf`

1. When did the Northside Community Garden Pilot begin?
2. How many raised beds does the garden have?
3. How many liters of water are used each week?
4. How many households does the project serve?
5. How much compost does the project produce each month?

An out-of-context question such as **“What is the capital of France?”** should receive the explicit not-found response rather than a guessed answer.

## Observations

- **Why AstraDB for production RAG?** It provides managed, scalable vector storage with persistence, metadata filtering, and hybrid search capabilities, avoiding the operational burden of hosting a vector index in each application process.
- **Why session state matters:** it keeps a user's conversation and UI state across Streamlit reruns. The vector database separately persists reusable document knowledge; session state is not a substitute for durable storage.
- **FAISS vs. AstraDB:** FAISS is an efficient local similarity-search library, often best for single-process experiments or self-managed deployments. AstraDB is a managed remote database designed for shared, persistent, multi-instance applications; it adds network latency, service configuration, and cloud cost.
- **Grounding limits:** retrieval and relevance thresholds reduce unsupported answers but cannot guarantee correctness. Review cited source pages for high-stakes use.

## Tests

```powershell
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Tests use fake AstraDB, embedding/model seams, and the bundled text-based sample PDF; they do not contact AstraDB, Hugging Face, or OpenAI. Live connection and model calls require your own credentials and an active AstraDB database. Image-only scanned PDFs need OCR before ingestion.

## Project structure

```text
.
├── data/
│   └── garden_report.pdf
├── src/
│   ├── astra_store.py
│   ├── config.py
│   ├── documents.py
│   ├── pdf_loader.py
│   ├── providers.py
│   └── rag.py
├── tests/
│   └── test_rag.py
├── app.py
├── main.py
├── requirements-test.txt
└── requirements.txt
```
