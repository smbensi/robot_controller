# Jetson Orin — arm64, JetPack 6.x (L4T r36)
# Base image: Python 3.10 on Ubuntu 22.04 (Jammy), same as JetPack 6
FROM python:3.10-slim-bookworm

# System deps for pymongo DNS (dnspython) and httpx
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install production dependencies first (layer cache friendly)
COPY requirements.txt ./
RUN pip install --no-cache-dir \
        aiomqtt \
        "pymongo[srv]" \
        httpx \
        "pydantic>=2.7.0" \
        "pydantic-settings>=2.3.0" \
        cachetools \
        structlog

# Copy source
COPY src/ ./src/
COPY data/ ./data/
COPY grammars/ ./grammars/

# Make src importable as a package
ENV PYTHONPATH=/app

# Default to JSON logs (readable by log collectors)
ENV APP_LOG_FORMAT=json

CMD ["python", "-m", "src.main"]
