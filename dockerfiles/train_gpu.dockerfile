FROM nvcr.io/nvidia/pytorch:24.12-py3

# --- system deps (rarely change) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# --- uv ---
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# --- dependency metadata FIRST (cache-friendly) ---
COPY pyproject.toml uv.lock ./

# --- install deps once, with cache ---
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# --- copy the rest (fast rebuilds) ---
COPY README.md .
COPY src/ src/
COPY .dvc/ .dvc/
COPY data_zip.dvc data_zip.dvc

# --- runtime config ---
RUN uv run dvc config core.no_scm true

ENTRYPOINT ["uv", "run", "python", "src/eurosat_classifier/scripts/entrypoint_train.py"]
