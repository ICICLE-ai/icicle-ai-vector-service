# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install . && \
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

# Drop to non-root user
USER app

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
