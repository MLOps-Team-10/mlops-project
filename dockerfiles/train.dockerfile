FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# cross-platform (Windows / Mac / Linux)
ENV HOME=/tmp
ENV UV_CACHE_DIR=/tmp/uv-cache

# Make the cache dir writable for any runtime UID (when using `docker run -u ...`)
RUN mkdir -p /tmp/uv-cache && chmod -R 777 /tmp/uv-cache

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY data/ data/

RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "python", "src/eurosat_classifier/scripts/entrypoint_train.py"]