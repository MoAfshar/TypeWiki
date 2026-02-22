# TypeWiki API Service Dockerfile

FROM python:3.12-slim AS builder

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install dependencies (no dev dependencies, frozen from lock file)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and install the project
COPY src/ ./src/
RUN uv sync --frozen --no-dev

FROM python:3.12-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY --from=builder /app/src ./src

# Copy PDF files (required for prompt building - lists available topics)
COPY pdfs/ ./pdfs/

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set Python to run unbuffered for better logging
ENV PYTHONUNBUFFERED=1

# Expose the API port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "typewiki.app:app", "--host", "0.0.0.0", "--port", "8000"]
