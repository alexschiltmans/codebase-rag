FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Ensure Python can find the codebase_rag package
ENV PYTHONPATH=/app/src

# Cache Hugging Face models on the persistent volume
ENV HF_HOME=/app/cache

# Create data directories
RUN mkdir -p data/repos data/cache cache logs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run via entrypoint (handles preloading + startup)
ENTRYPOINT ["/entrypoint.sh"]
CMD [".venv/bin/streamlit", "run", "src/codebase_rag/app/main.py", \
    "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
