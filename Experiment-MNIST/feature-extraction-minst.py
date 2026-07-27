
"""
Step 1: Feature Extraction with Frozen EfficientNet-B1 on MNIST

Outputs:
- features.npy  -> shape [70000, 1280]
- labels.npy    -> shape [70000]
- indices.npy   -> dataset indices (traceability)
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# -------------------------------
# Configuration
# -------------------------------
DATA_ROOT = "./data"
OUTPUT_DIR = "./outputs/mnist_features"
BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Preprocessing (paper-aligned)
# -------------------------------
transform = T.Compose([
    T.Resize((240, 240)),          # EfficientNet-B1 input
    T.Grayscale(num_output_channels=3),  # MNIST → RGB
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet
        std=[0.229, 0.224, 0.225],
    ),
])

# -------------------------------
# Dataset (downloadable)
# -------------------------------
train_ds = MNIST(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=transform,
)

test_ds = MNIST(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=transform,
)

dataset = torch.utils.data.ConcatDataset([train_ds, test_ds])

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# -------------------------------
# EfficientNet-B1 Feature Extractor
# -------------------------------
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.efficientnet_b1(weights="IMAGENET1K_V1")
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)  # [B, 1280]

extractor = EfficientNetFeatureExtractor().to(DEVICE)
extractor.eval()

# -------------------------------
# Feature Extraction Loop
# -------------------------------
all_features = []
all_labels = []
all_indices = []

idx_offset = 0

with torch.no_grad():
    for images, labels in tqdm(loader):
        images = images.to(DEVICE, non_blocking=True)
        feats = extractor(images)

        all_features.append(feats.cpu())
        all_labels.append(labels)
        all_indices.extend(range(idx_offset, idx_offset + len(labels)))
        idx_offset += len(labels)

features = torch.cat(all_features).numpy()
labels = torch.cat(all_labels).numpy()
indices = np.array(all_indices)

# -------------------------------
# Save to Disk
# -------------------------------
np.save(os.path.join(OUTPUT_DIR, "features.npy"), features)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
np.save(os.path.join(OUTPUT_DIR, "indices.npy"), indices)

print("====================================")
print("Feature extraction completed.")
print(f"Features shape : {features.shape}")
print(f"Labels shape   : {labels.shape}")
print(f"Saved to       : {OUTPUT_DIR}")
print("====================================")
