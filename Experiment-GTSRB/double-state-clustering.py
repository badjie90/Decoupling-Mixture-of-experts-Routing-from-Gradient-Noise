
"""
Input :
- outputs/gtsrb_features/features.npy
- outputs/gtsrb_split/labeled_idx.npy

Output:
- outputs/gtsrb_clusters/cluster_ids.npy
"""

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import davies_bouldin_score

IN_FEAT = "outputs/gtsrb_features/features.npy"
IN_SPLIT = "outputs/gtsrb_split/labeled_idx.npy"
OUT_DIR = "outputs/gtsrb_clusters"
os.makedirs(OUT_DIR, exist_ok=True)

X = np.load(IN_FEAT)
labeled_idx = np.load(IN_SPLIT)

K = 43
DIST_THR = 0.8

kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

dbi_before = davies_bouldin_score(X, labels)

nbrs = NearestNeighbors(n_neighbors=5).fit(X)
refined = labels.copy()

for i, x in enumerate(X):
    d = np.linalg.norm(x - centers[labels[i]])
    if d > DIST_THR:
        _, idx = nbrs.kneighbors([x])
        refined[i] = np.bincount(labels[idx[0]]).argmax()

dbi_after = davies_bouldin_score(X, refined)

np.save(f"{OUT_DIR}/cluster_ids.npy", refined)

print("GTSRB clustering done")
print(f"DBI before: {dbi_before:.4f} | after: {dbi_after:.4f}")
