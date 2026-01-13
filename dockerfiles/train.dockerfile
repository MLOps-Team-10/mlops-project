FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN chmod -R 777 /app

ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_PROJECT_ENVIRONMENT=/tmp/.venv

RUN mkdir -p /tmp/uv-cache /tmp/.venv && chmod -R 777 /tmp/uv-cache /tmp/.venv

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/

RUN uv sync --locked --no-cache --no-install-project

# IMPORTANT: uv sync may create subfolders with restrictive perms, fix them after sync
RUN chmod -R 777 /tmp/uv-cache /tmp/.venv

ENTRYPOINT ["uv", "run", "python", "src/eurosat_classifier/scripts/entrypoint_train.py"]