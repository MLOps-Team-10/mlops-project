FROM python:3.12-slim

WORKDIR /app

# ✅ system deps often needed by torch/torchvision
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
  && rm -rf /var/lib/apt/lists/*

COPY cloud_deploy/app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY cloud_deploy/app /app/app
COPY src /app/src
COPY models /app/models

ENV PYTHONPATH=/app/src
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
