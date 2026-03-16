import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# Set model to CPU
device = torch.device("cpu")

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model_path = r"D:\difficult airway assesment\Difficult_AirwayAssesment\data\model_intubation.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img_paths):
    imgs = [Image.open(p).convert('RGB').resize((224, 224)) for p in img_paths]
    collage = Image.new('RGB', (224 * 3, 224))
    for i, img in enumerate(imgs):
        collage.paste(img, (i * 224, 0))
    
    # Predict
    input_tensor = preprocess(collage).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).flatten()
    return probs.numpy()

# Sample 1 (Easy)
s1 = [
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\images\6.jpg",
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\images\5.jpg",
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\images\4.jpg"
]
# Sample 2 (Difficult)
s2 = [
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\data\difficult\diff1.jpg",
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\data\difficult\diff2.jpg",
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\data\difficult\diff3.jpg"
]

p1 = predict(s1)
p2 = predict(s2)

print(f"Sample 1: {p1}")
print(f"Sample 2: {p2}")
