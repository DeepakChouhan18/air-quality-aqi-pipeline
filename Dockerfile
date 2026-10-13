# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Use a slim Python base. We do NOT use alpine because psycopg2-binary
# requires glibc which alpine doesn't have by default.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only the packages needed to compile/install python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ml.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements-ml.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# WHY NON-ROOT USER:
# Running as root inside a container means a container escape gives an
# attacker root on the host. Using a dedicated app user limits the blast radius.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder stage (keeps runtime image slim)
COPY --from=builder /install /usr/local

# Copy only the source files the API needs
COPY src/ ./src/
COPY sql/ ./sql/

# Create directories the app writes to at runtime
RUN mkdir -p data/logs data/ml reports/figures && \
    chown -R appuser:appuser /app

USER appuser

# The port the FastAPI server listens on
EXPOSE 8000

# HEALTHCHECK: Docker will restart the container if /health returns non-200
# for 3 consecutive checks, giving Compose a built-in circuit breaker.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run uvicorn (production ASGI server). Workers=1 for predictability;
# increase for high-throughput deployments.
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
