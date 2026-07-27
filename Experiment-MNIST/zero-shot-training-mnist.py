
"""
Zero-shot SEAS-GMoE training (no class labels)
Dataset: MNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from scipy.stats import mode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = "outputs/mnist"
NUM_CLUSTERS = 10
EPOCHS = 20
BATCH = 256

# ---------------- Load Data ----------------
X = torch.tensor(np.load(f"{BASE}_features/features.npy")).float().to(DEVICE)
clusters = torch.tensor(
    np.load(f"{BASE}_clusters/cluster_ids.npy")
).long().to(DEVICE)

true_labels = np.load(f"{BASE}_features/labels.npy")  # evaluation only

loader = DataLoader(
    TensorDataset(X, clusters),
    batch_size=BATCH,
    shuffle=True,
)

# ---------------- Models ----------------
class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_CLUSTERS),
        )

    def forward(self, x):
        return self.net(x)

class Gating(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CLUSTERS),
        )

    def forward(self, x):
        return self.net(x)

experts = [Expert().to(DEVICE) for _ in range(NUM_CLUSTERS)]
gate = Gating().to(DEVICE)

optimizers = [
    torch.optim.Adam(e.parameters(), lr=1e-3) for e in experts
]
gate_opt = torch.optim.Adam(gate.parameters(), lr=1e-3)

# ---------------- Zero-shot Training ----------------
for epoch in range(EPOCHS):
    for xb, cb in loader:
        xb, cb = xb.to(DEVICE), cb.to(DEVICE)

        # Train experts on cluster IDs
        for k in cb.unique():
            mask = cb == k
            if mask.sum() == 0:
                continue
            optimizers[k].zero_grad()
            loss = F.cross_entropy(experts[k](xb[mask]), cb[mask])
            loss.backward()
            optimizers[k].step()

        # Train gating
        gate_opt.zero_grad()
        loss_gate = F.cross_entropy(gate(xb.detach()), cb)
        loss_gate.backward()
        gate_opt.step()

    print(f"[MNIST Zero-Shot] Epoch {epoch+1}/{EPOCHS}")

# ---------------- Zero-shot Evaluation ----------------
with torch.no_grad():
    cluster_pred = gate(X).argmax(1).cpu().numpy()

# Map clusters → labels
mapping = {}
for c in range(NUM_CLUSTERS):
    idx = cluster_pred == c
    if idx.sum() > 0:
        mapping[c] = mode(true_labels[idx], keepdims=False).mode

final_preds = np.array([mapping[c] for c in cluster_pred])
acc = accuracy_score(true_labels, final_preds)

print(f"MNIST Zero-Shot Accuracy: {acc:.4f}")
