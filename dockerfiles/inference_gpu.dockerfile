FROM  nvcr.io/nvidia/pytorch:24.12-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-cache --no-install-project

COPY src/ src/

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# evaluate container entrypoint
ENTRYPOINT ["uv", "run", "python", "-m", "eurosat_classifier.predict_folder"]
