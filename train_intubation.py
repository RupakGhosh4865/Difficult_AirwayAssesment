import os
import shutil
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from PIL import Image
from tqdm import tqdm
import joblib

# --- Data Augmentation Configuration ---
original_dir = r"D:\rupak try\airway\data"
augmented_dir = r"D:\rupak try\airway\data_augmented"
num_augmented_per_image = 3  # Number of new images to create per original

# --- Define augmentation pipeline ---
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15)
])

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# --- Copy and augment original images ---
if os.path.exists(augmented_dir):
    shutil.rmtree(augmented_dir)
shutil.copytree(original_dir, augmented_dir)

# --- Process each image ---
dataset = datasets.ImageFolder(original_dir)
print("Generating augmented images...")

for class_name, class_idx in dataset.class_to_idx.items():
    class_path = os.path.join(original_dir, class_name)
    output_class_path = os.path.join(augmented_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for fname in tqdm(os.listdir(class_path), desc=f"Augmenting '{class_name}'"):
        original_path = os.path.join(class_path, fname)
        try:
            img = Image.open(original_path).convert("RGB")
        except:
            continue  # skip corrupted images

        base_name, ext = os.path.splitext(fname)
        # Save augmented versions
        for i in range(num_augmented_per_image):
            aug_img = augmentation(img)
            aug_img = to_pil(to_tensor(aug_img))  # ensure correct format
            aug_name = f"{base_name}_aug{i+1}{ext}"
            aug_img.save(os.path.join(output_class_path, aug_name))

print("Data augmentation complete!")

# --- Training Configuration ---
BATCH_SIZE = 8

assert os.path.isdir(os.path.join(augmented_dir, "difficult")), "Missing 'difficult' folder!"
assert os.path.isdir(os.path.join(augmented_dir, "easy")), "Missing 'easy' folder!"

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
])

# --- Dataset ---
dataset = datasets.ImageFolder(augmented_dir, transform=transform)
class_names = dataset.classes
print("Classes:", class_names)

# --- Train/Val Split ---
num_total = len(dataset)
num_train = int(0.8 * num_total)
indices = np.random.permutation(num_total)
train_indices = indices[:num_train]
val_indices = indices[num_train:]

train_subset = torch.utils.data.Subset(dataset, train_indices)
val_subset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# --- Feature Extractor ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = torch.nn.Identity()  # remove final classification layer
resnet = resnet.to(device)
resnet.eval()

def extract_features(dataloader):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)
            feats = resnet(imgs).cpu().numpy()
            features.extend(feats)
            labels.extend(lbls.numpy())
    return np.array(features), np.array(labels)

# --- Extract features ---
print("Extracting features from training set...")
X_train, y_train = extract_features(train_loader)
print("Extracting features from validation set...")
X_val, y_val = extract_features(val_loader)

# --- Train SVM ---
print("Training SVM classifier...")
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# --- Evaluate ---
print("Evaluating model...")
y_pred = svm.predict(X_val)

acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, zero_division=0)
rec = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)
cm = confusion_matrix(y_val, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:\n", cm)

# --- Save ---
np.savez(os.path.join(augmented_dir, "svm_metrics.npz"),
         acc=acc, prec=prec, rec=rec, f1=f1)
np.savez(os.path.join(augmented_dir, "svm_confusion_matrix.npz"), cm=cm)

joblib.dump(svm, os.path.join(augmented_dir, "svm_model.pkl"))
with open(os.path.join(augmented_dir, "class_names.txt"), "w") as f:
    for c in class_names:
        f.write(c + "\n")

print("Training complete! SVM model and metrics saved.") 

