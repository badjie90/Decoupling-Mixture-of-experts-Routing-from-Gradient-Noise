
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

IN_FEAT = "outputs/gtsrb_features/features.npy"
IN_LABEL = "outputs/gtsrb_features/labels.npy"
IN_SPLIT = "outputs/gtsrb_split/labeled_idx.npy"
OUT_DIR = "outputs/gtsrb_siamese"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
MARGIN = 1.0

X = np.load(IN_FEAT)
y = np.load(IN_LABEL)
labeled_idx = np.load(IN_SPLIT)

X_l = X[labeled_idx]
y_l = y[labeled_idx]

# ---------------- Dataset ----------------
class PairDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x1 = self.X[idx]
        same = np.random.rand() > 0.5
        if same:
            idx2 = np.random.choice(np.where(self.y == self.y[idx])[0])
            label = 1
        else:
            idx2 = np.random.choice(np.where(self.y != self.y[idx])[0])
            label = 0
        return x1, self.X[idx2], torch.tensor(label, dtype=torch.float32)

# ---------------- Model ----------------
class SiameseNet(nn.Module):
    def __init__(self, in_dim=1280, emb_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x1, x2):
        return self.encoder(x1), self.encoder(x2)

def contrastive_loss(z1, z2, y):
    d = F.pairwise_distance(z1, z2)
    return torch.mean(y * d**2 + (1 - y) * torch.clamp(MARGIN - d, min=0)**2)

# ---------------- Training ----------------
ds = PairDataset(X_l, y_l)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

model = SiameseNet().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), LR)

for epoch in range(EPOCHS):
    model.train()
    total = 0
    for x1, x2, yb in dl:
        x1, x2, yb = x1.to(DEVICE), x2.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        z1, z2 = model(x1, x2)
        loss = contrastive_loss(z1, z2, yb)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss {total/len(dl):.4f}")

# ---------------- τ Selection ----------------
model.eval()
with torch.no_grad():
    emb = model.encoder(torch.tensor(X_l, device=DEVICE)).cpu()
    dist = torch.cdist(emb, emb).numpy()
    gt = (y_l[:, None] == y_l[None, :]).astype(int).flatten()
    d_flat = dist.flatten()

taus = np.linspace(d_flat.min(), d_flat.max(), 100)
best_tau, best_f1 = None, 0

for t in taus:
    pred = (d_flat <= t).astype(int)
    f1 = f1_score(gt, pred)
    if f1 > best_f1:
        best_f1, best_tau = f1, t

np.save(f"{OUT_DIR}/tau.npy", best_tau)
torch.save(model.encoder.state_dict(), f"{OUT_DIR}/siamese_encoder.pt")

print(f"GTSRB Siamese done | τ = {best_tau:.4f}")
