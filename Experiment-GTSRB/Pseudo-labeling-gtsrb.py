
import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------- Paths --------
FEAT = "outputs/gtsrb_features/features.npy"
LABEL = "outputs/gtsrb_features/labels.npy"
UNLAB = "outputs/gtsrb_split/unlabeled_idx.npy"
CLUST = "outputs/gtsrb_clusters/cluster_ids.npy"
ENC = "outputs/gtsrb_siamese/siamese_encoder.pt"
TAU = "outputs/gtsrb_siamese/tau.npy"
OUT = "outputs/gtsrb_pseudo"
os.makedirs(OUT, exist_ok=True)

PURITY_THR = 0.8

# -------- Load --------
X = np.load(FEAT)
y = np.load(LABEL)
unl_idx = np.load(UNLAB)
clusters = np.load(CLUST)
tau = float(np.load(TAU))

# -------- Encoder --------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        return self.net(x)

enc = Encoder().to(DEVICE)
enc.load_state_dict(torch.load(ENC))
enc.eval()

with torch.no_grad():
    emb = enc(torch.tensor(X, device=DEVICE)).cpu()

# -------- Nearest labeled neighbor --------
lab_mask = np.setdiff1d(np.arange(len(X)), unl_idx)
lab_emb = emb[lab_mask]
lab_y = y[lab_mask]

dist = torch.cdist(emb[unl_idx], lab_emb)
min_dist, nn_idx = dist.min(dim=1)

keep = min_dist.numpy() <= tau
pseudo_idx = unl_idx[keep]
pseudo_lbl = lab_y[nn_idx.numpy()[keep]]

# -------- Purity filtering --------
final_idx, final_lbl = [], []

for c in np.unique(clusters[pseudo_idx]):
    idxs = pseudo_idx[clusters[pseudo_idx] == c]
    lbls = pseudo_lbl[clusters[pseudo_idx] == c]
    maj, cnt = Counter(lbls).most_common(1)[0]
    if cnt / len(lbls) >= PURITY_THR:
        final_idx.extend(idxs)
        final_lbl.extend([maj] * len(idxs))

np.save(f"{OUT}/pseudo_idx.npy", np.array(final_idx))
np.save(f"{OUT}/pseudo_labels.npy", np.array(final_lbl))

print(f"GTSRB pseudo-labeling done | kept {len(final_idx)} samples")
