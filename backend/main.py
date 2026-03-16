import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import joblib

app = FastAPI()

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for deployment. Change this to specific URLs for better security.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Paths & Model Setup ---
# Use relative paths so it works on any server (Windows or Linux)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)  # Parent dir of backend/
DATA_DIR = os.path.join(BASE_DIR, "data")
AUGMENTED_DIR = os.path.join(BASE_DIR, "data_augmented")
MODEL_PATH = os.path.join(DATA_DIR, "model_intubation.pt")
CLASS_NAMES_PATH = os.path.join(DATA_DIR, "class_names.txt")
UTILS_DIR = os.path.join(BASE_DIR, "utils")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classes
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = [line.strip().capitalize() for line in f.readlines()]
else:
    CLASS_NAMES = ["Difficult", "Easy"]

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

MODEL = load_model()

@app.get("/")
async def root():
    return {"message": "Difficult Airway Assessment API"}

@app.post("/predict")
async def predict(
    neutral: UploadFile = File(...),
    tongue: UploadFile = File(...),
    headup: UploadFile = File(...)
):
    try:
        # Load images
        img1 = Image.open(io.BytesIO(await neutral.read())).convert('RGB')
        img2 = Image.open(io.BytesIO(await tongue.read())).convert('RGB')
        img3 = Image.open(io.BytesIO(await headup.read())).convert('RGB')

        # Create collage (same as Streamlit app)
        imgs_resized = [img.resize((224, 224)) for img in [img1, img2, img3]]
        collage = Image.new('RGB', (224 * 3, 224))
        for i, img in enumerate(imgs_resized):
            collage.paste(img, (i * 224, 0))

        # Run prediction
        input_tensor = preprocess(collage).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = MODEL(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
            pred_idx = int(np.argmax(probs))

        return {
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx] * 100),
            "probabilities": {
                "Easy": float(probs[1] * 100),
                "Difficult": float(probs[0] * 100)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    metrics_path = os.path.join(AUGMENTED_DIR, "svm_metrics.npz")
    cm_path = os.path.join(AUGMENTED_DIR, "svm_confusion_matrix.npz")
    
    data = {}
    if os.path.exists(metrics_path):
        m = np.load(metrics_path)
        data["accuracy"] = float(m["acc"])
        data["precision"] = float(m["prec"])
        data["recall"] = float(m["rec"])
        data["f1"] = float(m["f1"])
    
    if os.path.exists(cm_path):
        cm = np.load(cm_path)
        data["confusion_matrix"] = cm["cm"].tolist()
        
    return data

@app.get("/research-paper")
async def get_research_paper():
    # Find PDF in utils
    pdfs = [f for f in os.listdir(UTILS_DIR) if f.endswith('.pdf')]
    if not pdfs:
        raise HTTPException(status_code=404, detail="Research paper not found in utils folder")
    
    return FileResponse(os.path.join(UTILS_DIR, pdfs[0]), media_type='application/pdf')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
