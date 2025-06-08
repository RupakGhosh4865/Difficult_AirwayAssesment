# train_intubation.py

import os
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ---- Configuration ----
base_dir = r"D:\rupak try\data"
BATCH_SIZE = 8
EPOCHS = 10
WEIGHTS = "IMAGENET1K_V1"  # or 'DEFAULT' in new torchvision versions

assert os.path.isdir(os.path.join(base_dir, "difficult")), "Missing 'difficult' folder!"
assert os.path.isdir(os.path.join(base_dir, "easy")), "Missing 'easy' folder!"

# ---- Data transforms ----
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---- Dataset loading ----
dataset = datasets.ImageFolder(base_dir, transform=transform_train)
print("Class to idx mapping: ", dataset.class_to_idx)
print("Total images:", len(dataset))
for i in range(min(len(dataset.samples), 10)):
    print(f"{dataset.samples[i][0]}  -->  class {dataset.samples[i][1]}")

assert set(dataset.class_to_idx.keys()) == {"easy", "difficult"}, "Folders must be named 'easy' and 'difficult'."

# ---- Train/Val split (80/20 stratified) ----
num_train = int(0.8 * len(dataset))
indices = np.arange(len(dataset))
np.random.seed(42)  # reproducibility
np.random.shuffle(indices)
train_indices = indices[:num_train]
val_indices = indices[num_train:]

train_set = Subset(dataset, train_indices)
val_dataset_full = datasets.ImageFolder(base_dir, transform=transform_val)
val_set = Subset(val_dataset_full, val_indices)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# ---- Model ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---- Metric logs ----
val_acc_list = []
val_loss_list = []
val_prec_list = []
val_rec_list = []
val_f1_list = []

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    running_loss = 0.0
    running_correct = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        running_correct += (out.argmax(1) == y).sum().item()
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = running_correct / len(train_loader.dataset)

    # --- Validate ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            preds = out.argmax(1)
            val_correct += (preds == y).sum().item()
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)
    val_prec = precision_score(all_labels, all_preds, zero_division=0)
    val_rec = recall_score(all_labels, all_preds, zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)

    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)
    val_prec_list.append(val_prec)
    val_rec_list.append(val_rec)
    val_f1_list.append(val_f1)

    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc:.4f}; "
          f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

# --- Save metrics ---
np.savez(os.path.join(base_dir, "metrics.npz"),
         val_acc=np.array(val_acc_list),
         val_loss=np.array(val_loss_list),
         val_prec=np.array(val_prec_list),
         val_rec=np.array(val_rec_list),
         val_f1=np.array(val_f1_list))
print("Saved metrics.npz")

# --- Save confusion matrix on all validation data ---
cm = confusion_matrix(all_labels, all_preds)
np.savez(os.path.join(base_dir, "confusion_matrix.npz"), cm=cm)
print("Saved confusion_matrix.npz")

# --- Save model & class names ---
torch.save(model.state_dict(), os.path.join(base_dir, "model_intubation.pt"))
with open(os.path.join(base_dir, "class_names.txt"), "w") as f:
    for c in dataset.classes:
        f.write(c + "\n")

print("Training complete! Model, metrics, and confusion matrix saved to:", base_dir)