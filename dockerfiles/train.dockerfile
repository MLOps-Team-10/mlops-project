# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*



COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN uv sync --locked --no-cache --no-install-project

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

#ENV UV_LINK_MODE=copy
#RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENTRYPOINT ["uv", "run", "src/eurosat_classifier/train.py"]

#to search inside the image
#docker run --rm -it --entrypoint sh {image_name}:{image_tag}
