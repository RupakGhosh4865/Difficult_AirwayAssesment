import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

DEVICE = torch.device("cpu")
DATA_DIR = r"D:\difficult airway assesment\Difficult_AirwayAssesment\data"
MODEL_PATH = os.path.join(DATA_DIR, "model_intubation.pt")

def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def get_collage(img_paths):
    imgs = [Image.open(p).convert('RGB').resize((224, 224)) for p in img_paths]
    collage = Image.new('RGB', (224 * 3, 224))
    for i, img in enumerate(imgs):
        collage.paste(img, (i * 224, 0))
    return collage

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = load_model()

# Sample 1 (Presumably Easy)
s1_paths = [
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\images\6.jpg",
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\images\5.jpg",
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\images\4.jpg"
]
# Sample 2 (Difficult)
s2_paths = [
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\data\difficult\diff1.jpg",
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\data\difficult\diff2.jpg",
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\data\difficult\diff3.jpg"
]

for name, paths in [("Sample 1 (Easy)", s1_paths), ("Sample 2 (Difficult)", s2_paths)]:
    collage = get_collage(paths)
    input_tensor = preprocess(collage).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy().flatten()
    print(f"{name}: Index 0={probs[0]:.4f}, Index 1={probs[1]:.4f}")
