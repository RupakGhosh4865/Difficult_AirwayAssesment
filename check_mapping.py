import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# Use CPU
device = torch.device("cpu")

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model_path = r"D:\difficult airway assesment\Difficult_AirwayAssesment\data\model_intubation.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Preprocess (Standard ResNet)
def predict(img_paths):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    imgs = [Image.open(p).convert('RGB').resize((224, 224)) for p in img_paths]
    collage = Image.new('RGB', (224 * 3, 224))
    for i, img in enumerate(imgs):
        collage.paste(img, (i * 224, 0))
    
    # Resize the final collage to 224, 224 because that's what the training did
    final_img = collage.resize((224, 224))
    input_tensor = preprocess(final_img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).flatten().numpy()
    return probs

# Sample 1 (Easy)
s1 = [
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\images\6.jpg",
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\images\5.jpg",
    r"D:\difficult airway assesment\Difficult_AirwayAssesment\images\4.jpg"
]

res = predict(s1)
print(f"Sample 1 Prediction Probs: {res}")
