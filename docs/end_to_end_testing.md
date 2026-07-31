# End-to-End Testing Guide for Codebase RAG

How to set up and run end-to-end tests, covering both automated tests and manual checks.

## Prerequisites

- Docker and Docker Compose
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) installed natively and running. For a containerized one, `make services-start PROFILE=ollama` also needs `OLLAMA_BASE_URL=http://127.0.0.1:11435` in your `.env`: the container publishes on 11435, and tests pointed at the default 11434 find nothing there and skip.
- Chrome or Chromium browser (for UI testing)

## Environment Setup

All development tasks are managed through the `Makefile` in the project root. Run `make help` to see all available targets.

```bash
# Initial setup (creates venv, installs deps, generates .env)
make setup

# Start the required services (Qdrant, Langfuse)
make services-start
```

## Testing Workflow

### 1. Prepare Test Data

First, ingest test data into the vector database:

```bash
# Run data ingestion
make ingest
```

### 2. Run Automated Tests

The automated test suite can verify different aspects of the application:

```bash
# Run unit tests
make test-unit

# Run integration tests
make test-integration

# Run end-to-end tests (non-UI)
make test-e2e

# Run all tests (unit + integration + e2e)
make test
```

### 3. Start the Application

Start the Streamlit application:

```bash
make app
```

This will launch the Streamlit interface at http://localhost:8501.


### 4. Manual Testing

1. Open the application at http://localhost:8501
2. Make sure a codebase has been ingested.
3. Enter test queries such as:
   - "What is this codebase about?"
   - "How do I load a power grid network?"
   - "Show me an example of calculating power flow."
4. Verify that responses are accurate, source citations are provided, and code examples render correctly.

### 5. Cleanup

When you're done testing, clean up the environment:

```bash
# Stop services
make services-stop

# Clean up everything (removes containers and volumes)
make services-clean
```

## Troubleshooting

### Common Issues

#### Qdrant Connection Errors

If the application can't connect to Qdrant:

1. Check if the service is running: `make services-status`
2. Verify Qdrant is healthy: `curl http://localhost:6333/healthz`
3. Restart services: `make services-restart`

#### Ollama Issues

If Ollama isn't responding:

1. Verify Ollama is running: `curl "$OLLAMA_BASE_URL/api/version"`. This is `http://127.0.0.1:11434` for a native install, or `http://127.0.0.1:11435` for the containerized one (`make services-start PROFILE=ollama`/`PROFILE=full`).
2. Check if the model is downloaded: `ollama list` (containerized: `docker exec codebase-rag-ollama ollama list`)
3. Pull the model if needed: `ollama pull sam860/LFM2:350m` (containerized: `docker exec codebase-rag-ollama ollama pull sam860/LFM2:350m`)

#### Streamlit App Crashes

If the Streamlit app crashes:

1. Check the logs: `cat logs/app.log`
2. Verify the `.env` file has the correct configuration
3. Try running directly: `make app`
