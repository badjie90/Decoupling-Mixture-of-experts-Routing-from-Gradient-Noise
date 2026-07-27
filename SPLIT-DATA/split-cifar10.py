# scripts/04a_cifar10_split.py
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

IN_DIR = "outputs/cifar10_features"
OUT_DIR = "outputs/cifar10_split"
os.makedirs(OUT_DIR, exist_ok=True)

X = np.load(f"{IN_DIR}/features.npy")
y = np.load(f"{IN_DIR}/labels.npy")

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=42)
labeled_idx, unlabeled_idx = next(sss.split(X, y))

np.save(f"{OUT_DIR}/labeled_idx.npy", labeled_idx)
np.save(f"{OUT_DIR}/unlabeled_idx.npy", unlabeled_idx)

print("CIFAR-10 split done")
