FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-cache --no-install-project

COPY src/ src/

# evaluate container entrypoint
ENTRYPOINT ["uv", "run", "python", "-m", "eurosat_classifier.evaluate"]

#build with this command
#docker build -f dockerfiles/evaluate.dockerfile -t evaluate:latest .

#docker build -> creates an image
#docker run -> creates a container from the image and runs it

#running this command

#docker run --rm --name evaluate \
#  --shm-size=2g \
#  -v "$(pwd)/models/eurosat_best.pth:/models/eurosat_best.pth" \
#  -v "$(pwd)/data/raw/rgb:/data/raw/rgb" \
#  evaluate:latest \
#  --ckpt /models/eurosat_best.pth \
#  --data-dir /data/raw/rgb

#--rm remove the container after the command finishes
#--name give the container a name
#--shm-size increase shared memory size to avoid potential issues with large datasets
# "$(pwd)/models/eurosat_best.pth:/models/eurosat_best.pth" mount the model checkpoint from host to container
# "$(pwd)/data/raw/rgb:/data/raw/rgb" mount the data directory from host to container
# evaluate:latest specify the image to use
# --ckpt /models/eurosat_best.pth pass the model checkpoint argument to the evaluation script
# --data-dir /data/raw/rgb pass the data directory argument to the evaluation



