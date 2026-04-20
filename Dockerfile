# Wave backend — minimal image for Railway/container hosts.
# Does NOT bake in data/ or models/: those are downloaded from HF on first
# boot by app.backend.inference.download_artifacts_if_missing().

FROM python:3.11-slim

# System deps required by some Python wheels (sentence-transformers,
# torch CPU wheel for glibc, pillow, etc.). Keep minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps separately to maximize layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy code. scripts/ is needed because app.backend.inference imports the
# TwoTower class from scripts/train.py.
COPY app ./app
COPY scripts ./scripts

# Railway injects $PORT; fall back to 8000 locally.
ENV PORT=8000
EXPOSE 8000

# Shell form so $PORT expands at runtime.
CMD uvicorn app.backend.main:app --host 0.0.0.0 --port $PORT
