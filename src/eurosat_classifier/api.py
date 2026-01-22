import torch
import io
import os
from PIL import Image
from typing_extensions import Annotated
from torchvision import transforms
from eurosat_classifier.model import EuroSATModel, ModelConfig
from fastapi import FastAPI, UploadFile, File, Query
app = FastAPI()


MODEL_PATH = "models/eurosat_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

config = ModelConfig(
    model_name=checkpoint["model_name"],
    num_classes=checkpoint["num_classes"],
    pretrained=False,
)
model = EuroSATModel(config)

state_dict = checkpoint["state_dict"]

if state_dict and all(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.to(device)
model.eval()

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to Eurosat model API !!"}


@app.post("/takeaguess")
async def guess(
    file: Annotated[UploadFile, File(description="Upload satellite image")],
):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)


    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
                   "Industrial", "Pasture", "PermanentCrop", "Residential", 
                   "River", "SeaLake"]

    return {
        "guess": EUROSAT_CLASSES[int(top_idx.item())],
        "confidence": top_prob.item()
    }