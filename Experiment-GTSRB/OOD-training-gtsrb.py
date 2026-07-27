
"""
OOD training for SEAS-GMoE 
ID: GTSRB
OOD: SVHN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = "outputs/gtsrb"
NUM_EXPERTS = 43
EPOCHS = 10
LAMBDA_OOD = 0.5
BATCH = 256

# ---------------- Load ID Features ----------------
X_id = torch.tensor(
    np.load(f"{BASE}_features/features.npy")
).float().to(DEVICE)
clusters = torch.tensor(
    np.load(f"{BASE}_clusters/cluster_ids.npy")
).to(DEVICE)

# ---------------- OOD Dataset ----------------
transform = T.Compose([
    T.Resize((240, 240)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

svhn = SVHN(root="./data", split="test", download=True, transform=transform)
ood_loader = DataLoader(svhn, batch_size=BATCH, shuffle=True)

# ---------------- Feature Extractor ----------------
from torchvision.models import efficientnet_b1
backbone = efficientnet_b1(weights="IMAGENET1K_V1").features.to(DEVICE)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

def extract(x):
    with torch.no_grad():
        return backbone(x).mean(dim=[2, 3])

# ---------------- Gating Network ----------------
class Gating(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_EXPERTS),
        )
    def forward(self, x):
        return self.net(x)

gate = Gating().to(DEVICE)
gate.load_state_dict(torch.load(f"{BASE}_moe/gating.pt"))
opt = torch.optim.Adam(gate.parameters(), lr=1e-4)

# ---------------- OOD Training ----------------
for epoch in range(EPOCHS):
    for (xb_id, yb_id), (xb_ood, _) in zip(
        DataLoader(TensorDataset(X_id, clusters), BATCH, shuffle=True),
        ood_loader
    ):
        xb_id, yb_id = xb_id.to(DEVICE), yb_id.to(DEVICE)
        xb_ood = extract(xb_ood.to(DEVICE))

        # ID loss
        id_logits = gate(xb_id)
        loss_id = F.cross_entropy(id_logits, yb_id)

        # OOD loss (uniform)
        ood_logits = gate(xb_ood)
        ood_prob = F.softmax(ood_logits, dim=1)
        uniform = torch.full_like(ood_prob, 1 / NUM_EXPERTS)
        loss_ood = F.kl_div(ood_prob.log(), uniform, reduction="batchmean")

        loss = loss_id + LAMBDA_OOD * loss_ood

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"[GTSRB OOD] Epoch {epoch+1} | Loss {loss.item():.4f}")

torch.save(gate.state_dict(), f"{BASE}_moe/gating_ood.pt")
print("GTSRB OOD training complete")
