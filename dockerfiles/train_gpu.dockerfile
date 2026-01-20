FROM  nvcr.io/nvidia/pytorch:24.12-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*



WORKDIR /app


COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY .dvc/ .dvc/
COPY data_zip.dvc data_zip.dvc


RUN uv sync --locked --no-cache --no-install-project
RUN uv run dvc config core.no_scm true
# Uncomment the following lines to use UV cache for faster builds
#FROM base AS cached

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync


ENTRYPOINT ["uv", "run", "python", "src/eurosat_classifier/scripts/entrypoint_train.py"]
