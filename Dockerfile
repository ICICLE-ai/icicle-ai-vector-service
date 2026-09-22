# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY src/ ./src/

# Install the [rerank] extra from PyTorch's CPU wheel index. The default PyPI
# torch wheel bundles ~2.5GB of CUDA libraries that are pure dead weight on a
# CPU-only pod; the cpu index cuts the image roughly in half.
# Set RERANK=0 to build a slim image without cross-encoder support — the service
# still runs, and only the "cross_encoder" rerank method returns 503.
ARG RERANK=1

RUN if [ "$RERANK" = "1" ]; then \
        pip install --no-cache-dir --prefix=/install \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            ".[rerank]"; \
    else \
        pip install --no-cache-dir --prefix=/install .; \
    fi && \
    rm -rf /install/lib/python3.11/site-packages/setuptools* \
           /install/lib/python3.11/site-packages/wheel* \
           /install/lib/python3.11/site-packages/pip* \
           /install/lib/python3.11/site-packages/_distutils_hack*

# ---- Runtime stage ----
FROM python:3.11-slim AS runtime

# Security: run as non-root, remove build tools from base image
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/false --create-home app && \
    pip uninstall -y setuptools wheel pip 2>/dev/null; rm -rf /usr/lib/python3.11/ensurepip

# Copy only runtime packages from builder
COPY --from=builder /install/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /install/bin /usr/local/bin

# Copy application source
WORKDIR /home/app
COPY src/ ./src/

# Reranker weights are downloaded on first use and cached here. Pointing HF at a
# writable path under the app user avoids a read-only-home failure at runtime;
# mount a volume here to persist the download across pod restarts.
ENV HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_OFFLINE=0
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache

# Drop to non-root user
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

ENTRYPOINT ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
