# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM python:3.11-slim AS builder

# Set to 0 to build without cross-encoder reranking: the image drops from
# ~1.5GB to ~200MB, the service still runs, and only method="cross_encoder"
# returns 503.
ARG RERANK=1

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/

# Build into a venv at a fixed path. The runtime stage copies it verbatim to the
# same path, which keeps console-script shebangs (uvicorn) valid and avoids
# hand-deleting pip/setuptools out of site-packages.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Torch first, from the CPU wheel index ONLY.
#
# PyPI's torch wheel depends on the nvidia-cuda-* packages — roughly 2.5GB of
# GPU runtime that is dead weight on a CPU pod and dominates build time. The
# CPU index serves a build with those dependencies stripped.
#
# This must be a separate step *into the same venv*: pip decides a requirement
# is already satisfied by inspecting the environment it is installing into, so
# installing torch here means the next step sees torch>=2.2 as met and leaves it
# alone. (Installing with --prefix instead of a venv defeats that check, and pip
# silently pulls the CUDA build back in.)
RUN if [ "$RERANK" = "1" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch; \
    fi

RUN if [ "$RERANK" = "1" ]; then \
        pip install ".[rerank]"; \
    else \
        pip install .; \
    fi

# Fail the build rather than ship a CPU image carrying CUDA libraries. Without
# this, a dependency change that reintroduces the GPU wheel costs 2.5GB and
# several minutes of build time with no visible error.
RUN if pip list --format=freeze | grep -qi '^nvidia-'; then \
        echo "ERROR: CUDA packages present in a CPU-only image:"; \
        pip list --format=freeze | grep -i '^nvidia-'; \
        exit 1; \
    fi && \
    python -c "import sys; print('venv size check'); " && \
    du -sh /opt/venv

# ---- Runtime stage ----
FROM python:3.11-slim AS runtime

RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/false --create-home app

# Same path as the builder, so shebangs inside the venv resolve.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /home/app
COPY src/ ./src/

# Reranker weights are downloaded on first use and cached here. Pointing HF at a
# writable path under the app user avoids a read-only-home failure at runtime;
# mount a volume here to persist the download across pod restarts.
ENV HF_HOME=/home/app/.cache/huggingface
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

ENTRYPOINT ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
