
"""
Zero-shot evaluation for SEAS-GMoE
Dataset: MNIST
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = "outputs/mnist"
NUM_CLUSTERS = 10
USE_HUNGARIAN = True  # set False for majority vote

# ---------------- Load Data ----------------
X = torch.tensor(
    np.load(f"{BASE}_features/features.npy")
).float().to(DEVICE)

y_true = np.load(f"{BASE}_features/labels.npy")

# ---------------- Gating Network ----------------
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

gate = Gating().to(DEVICE)
gate.load_state_dict(torch.load(f"{BASE}_zeroshot/gating.pt"))
gate.eval()

# ---------------- Predict Clusters ----------------
with torch.no_grad():
    cluster_pred = gate(X).argmax(dim=1).cpu().numpy()

# ---------------- Cluster → Class Mapping ----------------
conf_mat = confusion_matrix(y_true, cluster_pred)

if USE_HUNGARIAN:
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
else:
    mapping = {
        c: np.bincount(y_true[cluster_pred == c]).argmax()
        for c in range(NUM_CLUSTERS)
        if np.any(cluster_pred == c)
    }

y_pred = np.array([mapping[c] for c in cluster_pred])

# ---------------- Metrics ----------------
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
cm = confusion_matrix(y_true, y_pred)

print("MNIST Zero-Shot Evaluation")
print(f"Accuracy  : {acc:.4f}")
print(f"Macro-F1  : {f1:.4f}")

np.save(f"{BASE}_zeroshot/confusion_matrix.npy", cm)
