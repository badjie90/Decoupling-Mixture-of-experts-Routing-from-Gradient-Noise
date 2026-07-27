
"""
OOD Evaluation for SEAS-GMoE
ID  : CIFAR-10
OOD : SVHN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import SVHN
from torchvision.models import efficientnet_b1
from torchvision import datasets as D, transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = "outputs/cifar10"
NUM_EXPERTS = 10
DATA_ROOT = "./data"

# ---------------- Load ID Data ----------------
X_id = torch.tensor(
    np.load(f"{BASE}_features/features.npy")
).float().to(DEVICE)

y_id = np.load(f"{BASE}_features/labels.npy")

# ---------------- Models ----------------
class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 43),
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
    m.load_state_dict(torch.load(f"{BASE}_moe/expert_{k}.pt"))
    m.eval()
    experts.append(m)

gate = Gating().to(DEVICE)
gate.load_state_dict(torch.load(f"{BASE}_moe/gating_ood.pt"))
gate.eval()

# ---------------- ID Inference ----------------
with torch.no_grad():
    gw = F.softmax(gate(X_id), dim=1)
    exp_out = torch.stack(
        [F.softmax(e(X_id), dim=1) for e in experts],
        dim=1
    )
    probs = (gw.unsqueeze(-1) * exp_out).sum(dim=1)
    preds = probs.argmax(dim=1).cpu().numpy()

acc = accuracy_score(y_id, preds)
f1 = f1_score(y_id, preds, average="macro")

# ---------------- OOD Dataset ----------------
transform = T.Compose([
    T.Resize((240, 240)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])





cifar10_ood = D.CIFAR10(
    root="./data",
    train=False,          # use test set for OOD evaluation (avoids train-set leakage)
    download=True,
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1] range
    ]),
)


# ---------------- Feature Extractor ----------------
backbone = efficientnet_b1(weights="IMAGENET1K_V1").features.to(DEVICE)
backbone.eval()

def extract(x):
    with torch.no_grad():
        return backbone(x).mean(dim=(2, 3))

# ---------------- Entropy Computation ----------------
def entropy(p):
    return -(p * torch.log(p + 1e-8)).sum(dim=1)

id_entropy = entropy(F.softmax(gate(X_id), dim=1)).cpu().numpy()

ood_entropy = []
for xb, _ in torch.utils.data.DataLoader(cifar10_ood, batch_size=256):
    xb = extract(xb.to(DEVICE))
    p = F.softmax(gate(xb), dim=1)
    ood_entropy.append(entropy(p).cpu())

ood_entropy = torch.cat(ood_entropy).numpy()

# ---------------- AUROC ----------------
labels = np.concatenate([
    np.zeros_like(id_entropy),
    np.ones_like(ood_entropy),
])

scores = np.concatenate([id_entropy, ood_entropy])
auroc = roc_auc_score(labels, scores)

print("CIFAR-10 OOD Evaluation")
print(f"ID Accuracy     : {acc:.4f}")
print(f"ID Macro-F1     : {f1:.4f}")
print(f"ID Entropy Mean : {id_entropy.mean():.4f}")
print(f"OOD Entropy Mean: {ood_entropy.mean():.4f}")
print(f"OOD AUROC       : {auroc:.4f}")
