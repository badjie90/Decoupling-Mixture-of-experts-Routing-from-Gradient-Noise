
"""
SEAS-GMoE Training with Optuna
Includes:
- Expert dataset construction (cluster-conditioned)
- Expert hyperparameter optimization
- Gating network hyperparameter optimization
- Joint gradient-decoupled fine-tuning
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- CONFIG ----------------
BASE = "outputs/cifar10"
NUM_CLASSES = 10
NUM_EXPERTS = 10
EXPERT_TRIALS = 15
GATE_TRIALS = 15
EPOCHS = 20

OUT = f"{BASE}_moe"
os.makedirs(OUT, exist_ok=True)

# ---------------- LOAD DATA ----------------
X = torch.tensor(np.load(f"{BASE}_features/features.npy")).float().to(DEVICE)
y = np.load(f"{BASE}_features/labels.npy")
clusters = np.load(f"{BASE}_clusters/cluster_ids.npy")

lab_idx = np.load(f"{BASE}_split/labeled_idx.npy")
p_idx = np.load(f"{BASE}_pseudo/pseudo_idx.npy")
p_lbl = np.load(f"{BASE}_pseudo/pseudo_labels.npy")

train_idx = np.concatenate([lab_idx, p_idx])
train_lbl = np.concatenate([y[lab_idx], p_lbl])
train_cluster = clusters[train_idx]

# ---------------- MODELS ----------------
class Expert(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)

class Gating(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_EXPERTS),
        )

    def forward(self, x):
        return self.net(x)

# ---------------- EXPERT OPTUNA ----------------
def expert_objective(trial, Xc, yc):
    hidden = trial.suggest_categorical("hidden", [256, 512, 768])
    drop = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    wd = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    bs = trial.suggest_categorical("batch", [64, 128, 256])

    Xtr, Xva, ytr, yva = train_test_split(Xc, yc, test_size=0.2, stratify=yc)

    model = Expert(hidden, drop).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    dl = DataLoader(
        TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr)),
        batch_size=bs, shuffle=True
    )

    for _ in range(EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()

    with torch.no_grad():
        pred = model(torch.tensor(Xva).float().to(DEVICE)).argmax(1).cpu()
    return (pred == torch.tensor(yva)).float().mean().item()

# ---------------- TRAIN EXPERTS ----------------
experts = []

for k in range(NUM_EXPERTS):
    sel = train_idx[train_cluster == k]
    if len(sel) < 50:
        experts.append(None)
        continue

    Xc, yc = X[sel].cpu().numpy(), train_lbl[train_cluster == k]

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: expert_objective(t, Xc, yc), n_trials=EXPERT_TRIALS)

    best = study.best_params
    model = Expert(best["hidden"], best["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=best["lr"],
        weight_decay=best["weight_decay"],
    )

    dl = DataLoader(
        TensorDataset(torch.tensor(Xc).float(), torch.tensor(yc)),
        batch_size=best["batch"], shuffle=True
    )

    for _ in range(EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()

    torch.save(model.state_dict(), f"{OUT}/expert_{k}.pt")
    json.dump(best, open(f"{OUT}/expert_{k}_params.json", "w"))
    experts.append(model)

# ---------------- GATING OPTUNA ----------------
def gate_objective(trial):
    hidden = trial.suggest_categorical("hidden", [128, 256, 512])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

    model = Gating(hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(X[train_idx], torch.tensor(train_cluster).to(DEVICE))
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    for _ in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()

    with torch.no_grad():
        pred = model(X[train_idx]).argmax(1).cpu()
    return (pred == torch.tensor(train_cluster)).float().mean().item()

gate_study = optuna.create_study(direction="maximize")
gate_study.optimize(gate_objective, n_trials=GATE_TRIALS)

gate = Gating(gate_study.best_params["hidden"]).to(DEVICE)
gate_opt = torch.optim.Adam(gate.parameters(), lr=gate_study.best_params["lr"])

# ---------------- JOINT FINE-TUNING ----------------
for _ in range(EPOCHS):
    for xb, yb, cb in DataLoader(
        TensorDataset(
            X[train_idx],
            torch.tensor(train_lbl).to(DEVICE),
            torch.tensor(train_cluster).to(DEVICE)
        ),
        batch_size=256,
        shuffle=True
    ):
        # Experts
        for k in torch.unique(cb):
            mask = cb == k
            if mask.sum() == 0:
                continue
            out = experts[k](xb[mask])
            F.cross_entropy(out, yb[mask]).backward()

        # Gating (gradient-decoupled)
        gate_opt.zero_grad()
        F.cross_entropy(gate(xb.detach()), cb).backward()
        gate_opt.step()

torch.save(gate.state_dict(), f"{OUT}/gating.pt")
print("CIFAR-10 SEAS-GMoE training complete")
