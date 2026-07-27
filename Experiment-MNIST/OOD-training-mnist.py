
"""
OOD training for SEAS-GMoE 
ID  : MNIST
OOD : Fashion-MNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import FashionMNIST
from torchvision.models import efficientnet_b1
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Configuration ----------------
BASE = "outputs/mnist"
NUM_EXPERTS = 10
EPOCHS = 10
BATCH_SIZE = 256
LAMBDA_OOD = 0.5
DATA_ROOT = "./data"

# ---------------- Load ID Features ----------------
X_id = torch.tensor(
    np.load(f"{BASE}_features/features.npy")
).float().to(DEVICE)

cluster_labels = torch.tensor(
    np.load(f"{BASE}_clusters/cluster_ids.npy")
).to(DEVICE)

id_loader = DataLoader(
    TensorDataset(X_id, cluster_labels),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# ---------------- OOD Dataset (Download) ----------------
transform = T.Compose([
    T.Resize((240, 240)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

ood_dataset = FashionMNIST(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=transform,
)

ood_loader = DataLoader(
    ood_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# ---------------- Feature Extractor ----------------
backbone = efficientnet_b1(weights="IMAGENET1K_V1").features.to(DEVICE)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

def extract_features(x):
    with torch.no_grad():
        x = backbone(x)
        return x.mean(dim=(2, 3))

# ---------------- Gating Network ----------------
class GatingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_EXPERTS),
        )

    def forward(self, x):
        return self.net(x)

gate = GatingNet().to(DEVICE)
gate.load_state_dict(torch.load(f"{BASE}_moe/gating.pt"))

optimizer = torch.optim.Adam(gate.parameters(), lr=1e-4)

# ---------------- OOD Training Loop ----------------
for epoch in range(EPOCHS):
    for (x_id, y_id), (x_ood, _) in zip(id_loader, ood_loader):
        x_id, y_id = x_id.to(DEVICE), y_id.to(DEVICE)
        x_ood = extract_features(x_ood.to(DEVICE))

        # ID routing loss
        logits_id = gate(x_id)
        loss_id = F.cross_entropy(logits_id, y_id)

        # OOD uniform routing loss
        logits_ood = gate(x_ood)
        probs_ood = F.softmax(logits_ood, dim=1)
        uniform = torch.full_like(probs_ood, 1.0 / NUM_EXPERTS)
        loss_ood = F.kl_div(probs_ood.log(), uniform, reduction="batchmean")

        loss = loss_id + LAMBDA_OOD * loss_ood

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[MNIST OOD] Epoch {epoch+1}/{EPOCHS} | Loss {loss.item():.4f}")

torch.save(gate.state_dict(), f"{BASE}_moe/gating_ood.pt")
print("MNIST OOD training completed.")
