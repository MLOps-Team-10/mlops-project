from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException

import os
import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

logger.info(f"Starting app - cwd = {os.getcwd()}")
logger.info(f"PYTHONPATH = {os.environ.get('PYTHONPATH')}")
logger.info(f"Files in /app: {os.listdir('/app')}")
logger.info(f"Files in /app/app: {os.listdir('/app/app') if os.path.exists('/app/app') else 'missing'}")
logger.info(f"PORT env = {os.environ.get('PORT', 'not set')}")
from eurosat_classifier.model import EuroSATModel, ModelConfig

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/eurosat_best.pth"))
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    cfg = ModelConfig(
        model_name="resnet18",
        num_classes=10,
        pretrained=False,
    )

    model = EuroSATModel(cfg)

    if not MODEL_PATH.exists():

        app.state.model = None
        app.state.model_error = f"Model file not found: {MODEL_PATH}"
        yield
        return

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    app.state.model = model
    app.state.model_error = None

    yield



app = FastAPI(lifespan=lifespan)

@app.get("/")
def health():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "model_path": str(MODEL_PATH),
        "model_error": app.state.model_error,
    }

@app.post("/predict")
def predict():
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail=app.state.model_error or "Model not loaded")

    return {"prediction": "AnnualCrop"}
