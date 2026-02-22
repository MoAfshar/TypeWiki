# TypeWiki
Copilot for Typeform's help center common questions

## Getting Started

### Prerequisites

Install [uv](https://github.com/astral-sh/uv), a fast Python package installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal.

**Note (macOS):** Check if `make` is installed:
```bash
make --version
```
If not available, install Xcode Command Line Tools:
```bash
xcode-select --install
```

### Installation

1. **Create a virtual environment:**
   ```bash
   make venv
   ```

2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Install the project with dev dependencies:**
   ```bash
   make install-dev-local
   ```

4. **Configure environment variables:**
   ```bash
   cp template.env .env
   ```
   Edit `.env` and set your values:
   - `AIRFLOW_HOME` - Path to the Airflow directory (defaults to `src/typewiki/airflow`)
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `OPENAI_MODEL_NAME` - The LLM model to use (e.g., `gpt-4o-mini`)
   - `OPENAI_EMBEDDING_MODEL_NAME` - Embedding model for RAG (e.g., `text-embedding-3-large`)
   - `PINECONE_API_KEY` - Your Pinecone API key for vector storage
   - `PINECONE_INDEX_NAME` - Name of your Pinecone index

5. **Export environment variables:**
   ```bash
   export $(cat .env | xargs)
   ```
   Run this command in your terminal session to load the environment variables. You'll need to run this each time you open a new terminal, or add it to your shell profile.

## Airflow Pipeline Setup

The project uses Apache Airflow to orchestrate the Help Center PDF ingestion pipeline.

### Initialize Airflow (One-Time)

After installing dependencies, initialize the Airflow database:

```bash
make airflow-init
```

### Adding More Help Center PDFs (Optional)

The project comes with sample PDFs pre-configured in `pdfs/` and `pdfs/pdf_manifest.json`. You can run the pipeline immediately with the existing articles.

If you want to add additional articles:

1. Place your PDF files in the `pdfs/` directory
2. Add entries to `pdfs/pdf_manifest.json` for each new PDF:

```json
{
  "articles": [
    {
      "filename": "your-article.pdf",
      "url": "https://help.typeform.com/hc/en-us/articles/...",
      "title": "Article Title",
      "category": "Category Name"
    }
  ]
}
```

### Populating the Vector Database

To ingest the Help Center articles into Pinecone, you need to run the ingestion pipeline. This processes the PDFs, chunks them, generates embeddings, and stores them in Pinecone.

**Option 1: Run via command line**
```bash
make airflow-ingest
```

**Option 2: Run via Airflow UI**
Start the Airflow web interface and manually trigger the DAG (see below).

### Running the Airflow Web UI

To start the full Airflow environment with the web interface:

```bash
make airflow-run
```

Access the UI at **http://localhost:8080**

**Login credentials:** When Airflow starts for the first time, it generates login credentials in `src/typewiki/airflow/simple_auth_manager_passwords.json.generated`. Check this file for the username and password to access the UI.

## Running the API Service

Once the vector database is populated, start the TypeWiki API service by running in the terminal:
```bash
typewiki
```
or 
```bash
make run
```

The API will be available at **http://localhost:8000**.

### API Endpoint

**POST /v1/chat**

```json
{
  "session_id": "unique-session-id",
  "message": "How do I create a multi-language form?",
  "history": []
}
```

## Make Commands Reference

| Command | Description |
|---------|-------------|
| `make venv` | Create a Python virtual environment |
| `make install-dev-local` | Install all dependencies for local development |
| `make test` | Run tests with coverage |
| `make lint` | Check code style with flake8 |
| `make clean` | Remove build, test, and Python artifacts |
| `make run` | Start the TypeWiki API service |

### Airflow Commands

| Command | Description |
|---------|-------------|
| `make airflow-init` | Initialize Airflow database (run once) |
| `make airflow-ingest` | Run the Help Center ingestion pipeline to populate Pinecone |
| `make airflow-list` | List all available DAGs |
| `make airflow-run` | Start Airflow web UI at http://localhost:8080 |
| `make airflow-clean` | Remove Airflow artifacts (database, logs) |

---

## Docker & Deployment

### Docker Setup

The project includes Docker configurations for both the API service and Airflow pipeline.

#### Building Docker Images

```bash
# Build the API service image
docker build -f docker/api/Dockerfile -t typewiki-api .

# Build the Airflow image
docker build -f docker/airflow/Dockerfile -t typewiki-airflow .
```

#### Running with Docker Compose

The easiest way to run both services locally is with Docker Compose:

```bash
# Start all services
docker-compose up

# Start only the API service
docker-compose up api

# Start only Airflow
docker-compose up airflow

# Stop all services
docker-compose down
```

**Prerequisites:** Ensure your `.env` file is configured with valid API keys before running.

- **API Service**: http://localhost:8000
- **Airflow UI**: http://localhost:8080

### Kubernetes Deployment

Helm charts are provided for deploying to Kubernetes clusters.

#### API Service Deployment

```bash
# Install the API service
helm install typewiki-api deploy/k8s/typewiki-api/ \
  --set secrets.openaiApiKey=<your-openai-key> \
  --set secrets.pineconeApiKey=<your-pinecone-key>

# Upgrade with new values
helm upgrade typewiki-api deploy/k8s/typewiki-api/ -f my-values.yaml

# Uninstall
helm uninstall typewiki-api
```

#### Airflow Ingestion Pipeline

```bash
# Install Airflow CronJob (runs daily at 2 AM by default)
helm install typewiki-airflow deploy/k8s/typewiki-airflow/ \
  --set secrets.openaiApiKey=<your-openai-key> \
  --set secrets.pineconeApiKey=<your-pinecone-key>

# Trigger manual ingestion
kubectl create job --from=cronjob/typewiki-airflow typewiki-airflow-manual
```

#### Production Considerations

For production Kubernetes deployments:

1. **Secrets Management**: Use external secret management (e.g., External Secrets Operator, HashiCorp Vault) instead of Helm-managed secrets
2. **Image Registry**: Push images to a container registry (ECR, GCR, Docker Hub) and update `image.repository` in values
3. **Ingress**: Enable and configure ingress for external access
4. **Airflow**: Consider using the [official Apache Airflow Helm Chart](https://airflow.apache.org/docs/helm-chart/stable/index.html) for production workloads with proper database backend

---

## Technical Architecture

### Overview

TypeWiki is a Retrieval-Augmented Generation (RAG) based chatbot designed to answer "How to..." questions from Typeform's Help Center documentation. The architecture consists of two main components:

1. **Ingestion Pipeline** - Processes PDF documents, chunks them, generates embeddings, and stores them in a vector database
2. **Chat API Service** - Handles user queries, retrieves relevant context, and generates responses using an LLM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TypeWiki Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│   │   PDF Files  │────▶│   Airflow    │────▶│   Pinecone   │                │
│   │  (Articles)  │     │   Pipeline   │     │ Vector Store │                │
│   └──────────────┘     └──────────────┘     └──────┬───────┘                │
│                                                    │                        │
│                                                    ▼                        │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│   │    User      │────▶│  Starlette   │────▶│   OpenAI     │                │
│   │   Request    │     │  API uvicorn │     │   LLM        │                │
│   └──────────────┘     └──────────────┘     └──────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | Starlette + Uvicorn | Lightweight async web framework with ASGI server |
| **Data Pipeline** | Apache Airflow | Workflow orchestration for document ingestion |
| **Vector Database** | Pinecone | Managed vector storage for semantic search |
| **LLM Provider** | OpenAI | Chat completion (GPT models) and embeddings |
| **LLM Framework** | LangChain | Unified interface for LLM operations and prompt management |
| **Data Validation** | Pydantic | Runtime data validation and settings management |
| **Package Manager** | uv | Fast Python dependency management |

### Project Structure

```
TypeWiki/
├── src/typewiki/              # Main application source code
│   ├── api.py                 # API endpoint definitions and request handling
│   ├── config.py              # Pydantic settings for environment configuration
│   ├── datamodels.py          # Request/response schemas (ChatRequest, ChatResponse)
│   ├── utils.py               # Starlette base class, endpoint decorator, logging
│   ├── exceptions.py          # Custom exception classes
│   ├── prompts/
│   │   └── copilot.py         # Prompt template and context formatting functions
│   └── airflow/
│       ├── airflow.cfg        # Airflow configuration
│       └── dags/
│           └── help_center_articles.py  # PDF ingestion DAG
├── docker/                    # Docker configurations
│   ├── api/Dockerfile         # API service container (multi-stage build)
│   └── airflow/Dockerfile     # Airflow pipeline container
├── deploy/k8s/                # Kubernetes Helm charts
│   ├── typewiki-api/          # API service Helm chart
│   └── typewiki-airflow/      # Airflow CronJob Helm chart
├── pdfs/                      # Help Center PDF articles
│   └── pdf_manifest.json      # Metadata for each PDF (title, URL, category)
├── docker-compose.yaml        # Local multi-container orchestration
├── Makefile                   # Development commands and shortcuts
├── pyproject.toml             # Project dependencies and metadata
├── template.env               # Environment variable template
└── README.md                  # This file
```

#### Key Files

| File | Responsibility |
|------|----------------|
| `api.py` | Defines `TypeWikiApp` class with `/v1/chat` endpoint. Initializes Pinecone vector store, OpenAI embeddings, and LLM on startup. Orchestrates the RAG flow: embed query → retrieve context → build prompt → generate response. |
| `config.py` | Type-safe configuration using Pydantic `BaseSettings`. Loads API keys and model names from environment variables with secure handling via `SecretStr`. |
| `datamodels.py` | Pydantic models for API contracts: `ChatRequest` (input), `ChatResponse` (output), and `ChatMessage` (history items). Provides runtime validation and serialization. |
| `prompts/copilot.py` | Contains the prompt template and helper functions to format retrieved context, conversation history, and available topics into the final prompt string. |
| `airflow/dags/help_center_articles.py` | Airflow DAG that reads PDFs, extracts text, chunks documents, generates embeddings, and upserts vectors to Pinecone. |
| `pdf_manifest.json` | Source of truth for available articles. Used by both the ingestion pipeline (to process PDFs) and the prompt builder (to list available topics). |

### Strategy Behind Technology Choices

**Starlette + Uvicorn over FastAPI:**
While FastAPI is more feature-rich, Starlette provides the minimal foundation needed for this API. The service has a single endpoint and doesn't require FastAPI's automatic documentation or dependency injection features. This choice keeps the codebase lean and dependencies minimal.

**Apache Airflow for Ingestion:**
Airflow provides robust workflow orchestration with built-in retry logic, monitoring, and scheduling capabilities. For a production system, the ingestion pipeline would run on a schedule (e.g., daily) to keep the knowledge base updated. Airflow's task-based architecture makes it easy to add new data sources or processing steps.

**Pinecone as Vector Database:**
Pinecone is a fully managed vector database that eliminates operational overhead. For a production RAG system, this is preferable to self-hosted solutions like Chroma or FAISS, which would require infrastructure management. Pinecone also offers excellent query performance and scales automatically.

**LangChain for LLM Operations:**
LangChain provides a unified interface for working with different LLM providers and embedding models. This abstraction makes it trivial to swap models or providers without changing application code. It also handles prompt templating and chain-of-thought patterns elegantly.

**Pydantic for Configuration and Validation:**
Pydantic's `BaseSettings` class provides type-safe configuration management with automatic environment variable loading. This ensures configuration errors are caught at startup rather than runtime, and sensitive values (like API keys) are handled securely with `SecretStr`.

---

## Design Decisions & Rationale

### Stateless Service Architecture

The TypeWiki API is **stateless by design**. The service does not store conversation history or user sessions. Instead, clients are expected to:

1. Maintain session state externally (in a backend service or client-side)
2. Pass relevant conversation history with each request via the `history` field

**Why this matters:**
- **Scalability**: Stateless services can be horizontally scaled without session affinity
- **Separation of Concerns**: Session management is a separate concern from Q&A functionality
- **Context Length Control**: The consuming service can decide how much history to include, preventing unbounded context growth
- **Flexibility**: Different clients can implement different history strategies (e.g., summarization, windowing)

### Prompt Engineering

The prompt template (`src/typewiki/prompts/copilot.py`) is aimed to deliver:

1. **Define scope explicitly** - Lists available topics so the model knows its knowledge boundaries
2. **Provide guardrails** - Clear instructions on when to admit limitations
3. **Structure context injection** - Retrieved documents are formatted with metadata for better grounding
4. **Include conversation history** - Enables multi-turn conversations while keeping the prompt self-contained

The prompt is the most critical component in a RAG system, it directly influences response quality, hallucination rates, and user experience.

### Vector Search Strategy

The current implementation uses **cosine similarity** for semantic search via Pinecone. This is a decent default for normalized embeddings.

**Alternative similarity metrics to explore:**
- **Euclidean Distance**: Better for some embedding spaces, especially if vectors aren't normalized
- **Maximal Marginal Relevance (MMR)**: Balances relevance with diversity to avoid redundant results

For production, implementing **hybrid search** (combining semantic search with MMR) could improve retrieval accuracy for queries with specific terminology.

### Chunking Strategy

Documents are chunked using a combination of:
- **RecursiveCharacterTextSplitter**: Respects document structure (paragraphs, sentences)
- **Overlap**: Adjacent chunks share content to preserve context across boundaries

The chunking strategy is critical because:
- Chunks that are too large dilute semantic signal and waste context window
- Chunks that are too small lose important context
- Poor chunk boundaries can split related information unnaturally

---

## Challenges & Solutions

### 1. Dynamic Web Scraping

**Challenge:** Scraping Help Center articles dynamically from the website proved difficult due to JavaScript rendering, anti-scraping measures, and inconsistent page structures.

**Solution:** Adopted a PDF-based approach using manual exports. This is pragmatic for the scope of this exercise but not ideal for production.

**Production Recommendation:** Integrate with a centralized content management system (CMS) or use Typeform's internal APIs to access Help Center content directly. This ensures data freshness and avoids scraping fragility.

### 2. Airflow Dependency Management

**Challenge:** Integrating Airflow into the project created dependency conflicts and increased installation complexity. Airflow has a large dependency footprint that can clash with other packages.

**Solution:** Configured Airflow with minimal dependencies and isolated its configuration in a dedicated directory.

**Production Recommendation:** Deploy Airflow as a separate service (e.g., Managed Airflow on AWS/GCP, or a dedicated Kubernetes deployment). This separates concerns and simplifies dependency management for the API service.

### 3. Embedding Model Latency

**Challenge:** Generating embeddings and LLM responses introduces latency. Large models like GPT-5 can take several seconds to respond.

**Solution:** Used async/await throughout the codebase and configured appropriate timeouts.

**Production Recommendation:** Consider smaller, faster models (e.g., GPT-4o-mini, Claude Haiku, or Gemini Flash) for latency-sensitive applications. Response streaming could also improve perceived performance.

---

## Future Improvements

### High Priority

1. **Add Guardrails**
   - Implement prompt injection detection to prevent malicious inputs
   - Add input validation for message length to prevent abuse
   - Rate limiting per session/IP

2. **Structured LLM Output**
   - Use Pydantic models with LangChain's structured output to ensure consistent response formats
   - Enable downstream services to programmatically process responses (e.g., extract action items, links)

3. **History Management**
   - Implement conversation summarization to compress long histories
   - Use sliding window approach to limit context growth
   - Consider semantic compression (keeping only relevant turns)

4. **Comprehensive Testing**
   - Unit tests for prompt building and context formatting
   - Integration tests for the RAG pipeline
   - End-to-end tests with mocked LLM responses
   - Evaluation framework for response quality

### Medium Priority

5. **Pipeline Improvements**
   - Add logic to detect and remove deleted articles from Pinecone
   - Implement incremental updates (only re-process changed documents)
   - Add content validation and deduplication

6. **Performance Optimization**
   - Implement response streaming for better UX
   - Cache common queries and their embeddings
   - Consider hedging strategies (query multiple providers, return fastest)

7. **Advanced Retrieval**
   - Implement hybrid search (semantic + keyword)
   - Add re-ranking step for retrieved documents
   - Experiment with different chunking strategies

### Nice to Have

8. **Observability**
   - Add tracing for LLM calls (e.g., LangSmith, MlFlow, OpenTelemetry)
   - Log retrieval metrics (relevance scores, number of results used)
   - Dashboard for query/business analytics (Grafana or Looker)

9. **Multi-Model Support**
   - Abstract LLM provider to support multiple backends

---

