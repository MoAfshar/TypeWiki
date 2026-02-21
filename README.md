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

## Airflow Pipeline Setup

The project uses Apache Airflow to orchestrate the Help Center PDF ingestion pipeline.

### Initialize Airflow (One-Time)

After installing dependencies, initialize the Airflow database:

```bash
make airflow-init
```

### Adding Help Center PDFs

1. Place your PDF files in the `pdfs/` directory
2. Update `pdfs/pdf_manifest.json` with metadata for each PDF:

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

### Testing the Pipeline

Run the DAG to test that everything works:

```bash
make airflow-test
```

### Running the Airflow Web UI

To start the full Airflow environment with the web interface:

```bash
make airflow-run
```

Access the UI at **http://localhost:8080**

**Login credentials:** When Airflow starts for the first time, it generates login credentials in `src/typewiki/airflow/simple_auth_manager_passwords.json`. Check this file for the username and password to access the UI.

## Make Commands Reference

| Command | Description |
|---------|-------------|
| `make venv` | Create a Python virtual environment |
| `make install-dev-local` | Install all dependencies for local development |
| `make test` | Run tests with coverage |
| `make lint` | Check code style with flake8 |
| `make clean` | Remove build, test, and Python artifacts |

### Airflow Commands

| Command | Description |
|---------|-------------|
| `make airflow-init` | Initialize Airflow database (run once) |
| `make airflow-test` | Test the Help Center ingest DAG |
| `make airflow-list` | List all available DAGs |
| `make airflow-run` | Start Airflow web UI at http://localhost:8080 |
| `make airflow-clean` | Remove Airflow artifacts (database, logs) |
