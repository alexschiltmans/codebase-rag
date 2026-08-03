#!/bin/bash
# Entrypoint script for the Codebase RAG Docker container.
# Auto-ingestion of the default repo is handled by the Streamlit UI.
set -e

# Wait for Qdrant to be ready
QDRANT_URL="http://${QDRANT_HOST:-localhost}:${QDRANT_PORT:-6333}"
echo "Waiting for Qdrant at ${QDRANT_URL}..."
QDRANT_READY=0
for i in $(seq 1 30); do
    if curl -sf "${QDRANT_URL}/healthz" > /dev/null 2>&1; then
        echo "Qdrant is ready."
        QDRANT_READY=1
        break
    fi
    sleep 2
done
if [ $QDRANT_READY -eq 0 ]; then
    echo "WARNING: Qdrant did not respond after 60 seconds."
fi

# The Ollama wait/pull/verify dance only makes sense when the app is actually
# going to talk to Ollama. On LLM_PROVIDER=openai-compat, MODEL is a model id
# on some other server (LM Studio, vLLM, ...) that Ollama can never have, so
# running this block unconditionally means every start tries to pull a model
# Ollama will never find and prints a warning naming an Ollama command that
# fixes nothing.
if [ "${LLM_PROVIDER:-ollama}" = "ollama" ]; then
    # Wait for Ollama to be ready
    MODEL="${LLM_MODEL_NAME:-sam860/LFM2:350m}"
    OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
    echo "Waiting for Ollama at ${OLLAMA_URL}..."
    OLLAMA_READY=0
    for i in $(seq 1 30); do
        if curl -sf "${OLLAMA_URL}/api/version" > /dev/null 2>&1; then
            echo "Ollama is ready."
            OLLAMA_READY=1
            break
        fi
        sleep 2
    done
    if [ $OLLAMA_READY -eq 0 ]; then
        echo "WARNING: Ollama did not respond after 60 seconds."
    fi

    # Pull default model if needed
    echo "Ensuring model '${MODEL}' is available in Ollama..."
    curl -s "${OLLAMA_URL}/api/pull" -d "{\"name\": \"${MODEL}\"}" > /dev/null 2>&1 || true

    # Verify model was pulled successfully
    case "${MODEL}" in
        *:*) MODEL_TAG_PATTERN="\"name\":\"${MODEL}\"" ;;
        *) MODEL_TAG_PATTERN="\"name\":\"${MODEL}(:latest)?\"" ;;
    esac
    # Same backend split the Python client makes: an Ollama-compatible endpoint is not always Ollama.
    case "${OLLAMA_URL}" in
        *//ollama:*|*:11435*) PULL_COMMAND="docker exec codebase-rag-ollama ollama pull ${MODEL}" ;;
        *:12434*|*model-runner.docker.internal*) PULL_COMMAND="docker model pull ${MODEL}" ;;
        *) PULL_COMMAND="ollama pull ${MODEL}" ;;
    esac

    if ! curl -s "${OLLAMA_URL}/api/tags" | grep -qE "${MODEL_TAG_PATTERN}"; then
        cat << EOF
================================================================================
WARNING: Model '${MODEL}' not found in Ollama
================================================================================
The configured model is not available. To pull it manually, run:

    ${PULL_COMMAND}

Then restart the app. The check will refresh on app restart.
================================================================================
EOF
    fi
else
    echo "LLM_PROVIDER=${LLM_PROVIDER}: skipping Ollama wait/pull, model lives on the configured backend instead."
fi

# Run whichever process the container was configured with (image CMD or a
# service-level command:), now that the shared dependency waits above are done.
exec "$@"
