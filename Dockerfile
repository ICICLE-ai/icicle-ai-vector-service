# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install .

# ---- Runtime stage ----
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/false --create-home app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
WORKDIR /home/app
COPY src/ ./src/

# Drop to non-root user
USER app

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
