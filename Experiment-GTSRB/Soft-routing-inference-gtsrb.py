# scripts/07_infer_eval_gtsrb.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Paths --------
BASE = "outputs/gtsrb"
FEAT = f"{BASE}_features/features.npy"
LABEL = f"{BASE}_features/labels.npy"
MOE = f"{BASE}_moe"
OUT = f"{BASE}_eval"
os.makedirs(OUT, exist_ok=True)

NUM_CLASSES = 43
NUM_EXPERTS = 43

# -------- Load Data --------
X = torch.tensor(np.load(FEAT)).float().to(DEVICE)
y = np.load(LABEL)

# -------- Models --------
class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASSES),
        )
    def forward(self, x):
        return self.net(x)

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

experts = []
for k in range(NUM_EXPERTS):
    m = Expert().to(DEVICE)
    m.load_state_dict(torch.load(f"{MOE}/expert_{k}.pt"))
    m.eval()
    experts.append(m)

gate = Gating().to(DEVICE)
gate.load_state_dict(torch.load(f"{MOE}/gating.pt"))
gate.eval()

# -------- Soft Routing Inference --------
with torch.no_grad():
    gate_w = F.softmax(gate(X), dim=1)           # [N, K]
    expert_out = torch.stack(
        [F.softmax(e(X), dim=1) for e in experts],
        dim=1                                    # [N, K, C]
    )
    probs = (gate_w.unsqueeze(-1) * expert_out).sum(dim=1)
    preds = probs.argmax(dim=1).cpu().numpy()

# -------- Evaluation --------
acc = accuracy_score(y, preds)
f1 = f1_score(y, preds, average="macro")
cm = confusion_matrix(y, preds)

np.save(f"{OUT}/confusion_matrix.npy", cm)

print("GTSRB Soft Routing Evaluation")
print(f"Accuracy : {acc:.4f}")
print(f"Macro-F1 : {f1:.4f}")
