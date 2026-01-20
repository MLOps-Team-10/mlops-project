import torch
import io
from fastapi import FastAPI, UploadFile, File, Query
from PIL import Image
from torchvision import transforms
from eurosat_classifier.model import EuroSATModel, ModelConfig
app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("models/eurosat_models_eurosat_best.zip", map_location=device)

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/")
async def health():
    return {"status": "ok", "model": checkpoint["model_name"], "num_classes": checkpoint["num_classes"]}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    max_length: int = Query(default=10, description="hyperparameter")
):
    # 3. Read and Preprocess Image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        logits = model(tensor)
        prediction = logits.argmax(dim=1).item()

    #Mapping indices to class names
    class_names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
                   "Industrial", "Pasture", "PermanentCrop", "Residential", 
                   "River", "SeaLake"]

    return {
        "filename": file.filename,
        "prediction": class_names[prediction],
        "class_index": prediction,
        "meta_max_length": max_length
    }