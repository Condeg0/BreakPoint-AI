# ---------------------------------------------------------
# Build Stage: Dependency Resolution
# ---------------------------------------------------------
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# ---------------------------------------------------------
# Production Stage: Minimal Runtime
# ---------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN addgroup --system mlopsgroup && adduser --system --group mlopsuser

WORKDIR /app

# Copy compiled wheels from builder
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy operational code and configuration
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY model_prod/ /app/artifacts/prod/

# NEW: Embed the historical ledger for batch context
COPY data/raw/ /app/data/raw/

RUN chown -R mlopsuser:mlopsgroup /app
USER mlopsuser

EXPOSE 8080

CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080}