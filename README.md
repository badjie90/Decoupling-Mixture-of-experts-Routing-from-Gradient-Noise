# SEAS-GMoE: Decoupling-Mixture-of-experts-Routing-from-Gradient-Noise
#### Decoupling Mixture-of-experts Routing from Gradient Noise: A Framework for Structured Specialization and Soft Generalization Toward Robust and Efficient Inference




# SEAS-GMoE: Semi-Supervised, Zero-Shot, and OOD-Aware Mixture of Experts

This repository contains an experimental end-to-end implementation of a
feature-space Mixture-of-Experts (MoE) pipeline for three image-classification
benchmarks:

- **MNIST** — 10 handwritten-digit classes;
- **CIFAR-10** — 10 natural-image classes;
- **GTSRB** — 43 German traffic-sign classes.

The project combines frozen EfficientNet-B1 feature extraction, a small
labeled-data split, double-state clustering, Siamese metric learning,
pseudo-label generation, cluster-specialized experts, learned soft routing,
zero-shot routing experiments, and out-of-distribution (OOD) training and
evaluation.

This is a single, unified project. All dataset pipelines share the same overall
design and write their artifacts into a common `outputs/` directory.


## Contents

- [Project overview](#project-overview)
- [Pipeline](#pipeline)
- [Repository structure](#repository-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [Complete dataset workflows](#complete-dataset-workflows)
- [Zero-shot experiments](#zero-shot-experiments)
- [OOD-aware routing](#ood-aware-routing)
- [Output artifacts](#output-artifacts)
- [Reproducibility](#reproducibility)
- [Publishing to GitHub](#publishing-to-github)
- [Citation and license](#citation-and-license)

## Project overview

The main supervised/semi-supervised path works entirely on 1,280-dimensional
features extracted by a frozen ImageNet-pretrained EfficientNet-B1 backbone.
Only the feature-extraction and OOD scripts process images directly; clustering,
Siamese training, pseudo-labeling, expert training, routing, and ordinary
in-distribution evaluation operate on saved NumPy features.

For a dataset with `C` classes, the current design generally uses `C` clusters
and `C` experts:

| Dataset | Classes | Intended clusters | Intended experts | OOD dataset |
|---|---:|---:|---:|---|
| MNIST | 10 | 10 | 10 | Fashion-MNIST |
| CIFAR-10 | 10 | 10 | 10 | SVHN |
| GTSRB | 43 | 43 | 43 | SVHN |

The baseline pipeline is semi-supervised: only 10% of samples are placed in the
initial labeled subset, while the remaining 90% are candidates for
pseudo-labeling. The split is stratified with random seed 42.

## Pipeline

```text
Raw torchvision dataset
        │
        ▼
Frozen EfficientNet-B1 feature extraction
        │
        ├── features.npy  [N, 1280]
        ├── labels.npy    [N]
        └── indices.npy   [N]
        │
        ▼
Stratified labeled/unlabeled split (10% / 90%)
        │
        ├───────────────┐
        ▼               ▼
Double-state       Siamese network
clustering         trained on labeled pairs
        │               │
        │               ├── siamese_encoder.pt
        │               └── tau.npy
        └───────┬───────┘
                ▼
        Cluster-aware pseudo-labeling
                │
                ├── pseudo_idx.npy
                └── pseudo_labels.npy
                │
                ▼
      Cluster-specialized expert training
      + Optuna gate optimization
                │
                ├── expert_<k>.pt
                ├── expert_<k>_params.json
                └── gating.pt
                │
         ┌──────┴────────┐
         ▼               ▼
Soft-routing       OOD-aware gate training
inference                │
                        ├── gating_ood.pt
                        └── OOD evaluation
```

The zero-shot branch is separate: it trains experts and a gate using cluster IDs
instead of semantic class labels, then maps predicted clusters to classes for
evaluation. That branch currently needs a checkpoint-saving correction described
below.

## Repository structure

```text
PRODUCTION/
├── README.md
├── MNIST/
│   ├── feature-extraction-minst.py
│   ├── double-state-clustering-mnist.py
│   ├── SNN-training-MNIST.py
│   ├── Pseudo-labeling-mnist.py
│   ├── moe-training-mnist.py
│   ├── Soft-routing-inference-mnist.py
│   ├── zero-shot-training-mnist.py
│   ├── zero-shot-evaluation-mnist.py
│   ├── OOD-training-mnist.py
│   └── OOD-evaluation-mnist.py
├── CIFAR-10/
│   ├── ferature-extraction-cifar10.py
│   ├── double-state-clustering-cifar10.py
│   ├── SNN-training-cifar10.py
│   ├── Pseudo-labeling-cifar10.py
│   ├── moe-training-cifar10.py
│   ├── Soft-routing-inference-cifar10.py
│   ├── zero-shot-training-cifar10.py
│   ├── zero-shot-evaluation-cifar10.py
│   ├── OOD-training-cifar10.py
│   └── OOD-evaluation-cifar10.py
├── GTSRB/
│   ├── feature-extraction.py
│   ├── double-state-clustering.py
│   ├── SNN-training-gtsrb.py
│   ├── Pseudo-labeling-gtsrb.py
│   ├── moe-training-gtsrb.py
│   ├── Soft-routing-inference-gtsrb.py
│   ├── zero-shot-training-gtsrb.py
│   ├── zero-shot-evaluation-gtsrb.py
│   ├── OOD-training-gtsrb.py
│   └── OOD-evaluation-gtsrb.py
└── SPLIT-DATA/
    ├── split-mnist.py
    ├── split-cifar10.py
    └── split-gtsrb.py
```

Some filenames contain spelling or capitalization inconsistencies. Use the exact
names shown above, including:

- `feature-extraction-minst.py` (`minst`, not `mnist`);
- `ferature-extraction-cifar10.py` (`ferature`, not `feature`);
- `SNN-training-MNIST.py` (uppercase `MNIST`).

## Script responsibilities

### Feature extraction

The three feature-extraction scripts download the official training and test
splits through torchvision, concatenate them, resize images for EfficientNet-B1,
and save frozen 1,280-dimensional embeddings.

### Data splitting

The scripts in `SPLIT-DATA/` create stratified labeled and unlabeled index files.
The current setting uses `test_size=0.9`, which means 10% labeled and 90%
unlabeled.

### Double-state clustering

These scripts apply K-means and then refine assignments using a nearest-neighbor
distance rule. They report the Davies-Bouldin index before and after refinement
and save `cluster_ids.npy`.

### Siamese neural network training

The SNN scripts learn a feature-space encoder from labeled positive and negative
pairs using a contrastive objective. They select and save a similarity/distance
threshold `tau.npy` together with `siamese_encoder.pt`.

### Pseudo-labeling

Pseudo-labeling loads unlabeled features, cluster assignments, the learned
Siamese encoder, and threshold. Candidate labels are filtered by the learned
distance rule and a cluster-purity threshold of `0.8`.

### MoE training

The MoE scripts combine genuinely labeled samples with accepted pseudo-labels.
They use Optuna to tune each cluster-conditioned expert and the gate, then save
expert weights, expert hyperparameters, and `gating.pt`.

### Soft-routing inference

Soft inference loads all experts and the gate. Gate softmax probabilities weight
the expert class distributions, producing the final prediction. Accuracy,
macro-F1, and a confusion matrix are reported.

### Zero-shot experiments

Zero-shot training uses cluster IDs rather than semantic labels. Evaluation maps
clusters to true labels, optionally through Hungarian assignment, so semantic
labels are still used to score the result.

### OOD training and evaluation

OOD training starts from `gating.pt`, keeps ordinary cluster-routing loss on
in-distribution features, and adds a KL-divergence objective that encourages
uniform routing for OOD samples. Evaluation uses routing entropy as the OOD
score and reports ID accuracy, ID macro-F1, mean entropies, and OOD AUROC.

## Datasets

No manual dataset download is normally required. The scripts instantiate
torchvision datasets with `download=True` and store them under `./data`.

- [MNIST torchvision documentation](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
- [CIFAR-10 dataset information](https://www.cs.toronto.edu/~kriz/cifar.html)
- [GTSRB benchmark website](https://benchmark.ini.rub.de/gtsrb_dataset.html)
- [torchvision datasets documentation](https://docs.pytorch.org/vision/stable/datasets.html)

The OOD branches additionally download:

- Fashion-MNIST for MNIST OOD training/evaluation;
- SVHN test data for CIFAR-10 OOD training/evaluation;
- SVHN test data for GTSRB OOD training/evaluation.

The first feature-extraction or OOD run needs internet access both for dataset
archives and for the pretrained EfficientNet-B1 weights. Later runs reuse local
caches.

## Installation

Python 3.10 or newer is recommended. GPU acceleration is strongly recommended,
especially for feature extraction, Optuna expert searches, and GTSRB's 43-expert
configuration.

Create an environment from the `PRODUCTION/` directory:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch and torchvision using the command appropriate for your CPU or
CUDA environment from the [official PyTorch
installer](https://pytorch.org/get-started/locally/). Then install the remaining
packages:

```bash
python -m pip install numpy scipy scikit-learn optuna tqdm
```

A suitable `requirements.txt` would contain:

```text
numpy>=1.26
scipy>=1.11
scikit-learn>=1.4
optuna>=3.5
tqdm>=4.66
torch>=2.1
torchvision>=0.16
```

PyTorch and torchvision versions must be mutually compatible. If CUDA is used,
the PyTorch build must also match the installed NVIDIA driver/runtime.

## Configuration

The scripts do not expose command-line arguments. Configuration is defined by
module-level constants near the top of each file. Edit these constants before
running when the defaults do not fit your machine:

- `DEVICE` — CPU or CUDA device;
- `DATA_ROOT` — downloaded dataset directory, normally `./data`;
- `BASE`, `IN_DIR`, `OUT_DIR`, `OUTPUT_DIR` — artifact paths;
- `BATCH_SIZE` or `BATCH` — memory/performance tradeoff;
- `NUM_WORKERS` — data-loading processes;
- `EPOCHS` — training duration;
- `K`, `NUM_CLUSTERS`, `NUM_EXPERTS`, `NUM_CLASSES` — dataset dimensions;
- `EXPERT_TRIALS`, `GATE_TRIALS` — Optuna search cost;
- `PURITY_THR`, `DIST_THR`, `LAMBDA_OOD` — method thresholds/weights.

All paths are relative to the current working directory. **Run every command
from the `PRODUCTION/` root**, not from inside `MNIST/`, `CIFAR-10/`, or `GTSRB/`.
Otherwise the scripts will read and write different `data/` and `outputs/`
directories.

Several scripts request `cuda:0`, `cuda:1`, or `cuda:2` explicitly. If your
machine has only one GPU, change those constants to `cuda:0` or use `cpu`.

## Quick start

The smallest ordinary end-to-end pipeline is MNIST. After addressing the known
issues relevant to your intended experiment, run:

```bash
cd /path/to/PRODUCTION

python MNIST/feature-extraction-minst.py
python SPLIT-DATA/split-mnist.py
python MNIST/double-state-clustering-mnist.py
python MNIST/SNN-training-MNIST.py
python MNIST/Pseudo-labeling-mnist.py
python MNIST/moe-training-mnist.py
python MNIST/Soft-routing-inference-mnist.py
```

The first command downloads MNIST and EfficientNet-B1 weights. Each later stage
depends on the files generated by the preceding stages.

## Complete dataset workflows

### MNIST

Run the core semi-supervised MoE pipeline in this exact order:

```bash
python MNIST/feature-extraction-minst.py
python SPLIT-DATA/split-mnist.py
python MNIST/double-state-clustering-mnist.py
python MNIST/SNN-training-MNIST.py
python MNIST/Pseudo-labeling-mnist.py
python MNIST/moe-training-mnist.py
python MNIST/Soft-routing-inference-mnist.py
```

Expected main artifact directories:

```text
outputs/mnist_features/
outputs/mnist_split/
outputs/mnist_clusters/
outputs/mnist_siamese/
outputs/mnist_pseudo/
outputs/mnist_moe/
outputs/mnist_eval/
```

### CIFAR-10

```bash
python CIFAR-10/ferature-extraction-cifar10.py
python SPLIT-DATA/split-cifar10.py
python CIFAR-10/double-state-clustering-cifar10.py
python CIFAR-10/SNN-training-cifar10.py
python CIFAR-10/Pseudo-labeling-cifar10.py
python CIFAR-10/moe-training-cifar10.py
python CIFAR-10/Soft-routing-inference-cifar10.py
```

Before the last command, correct the CIFAR soft-inference `BASE` constant as
described in [Known limitations](#known-limitations-and-required-corrections).

Expected main artifact directories:

```text
outputs/cifar10_features/
outputs/cifar10_split/
outputs/cifar10_clusters/
outputs/cifar10_siamese/
outputs/cifar10_pseudo/
outputs/cifar10_moe/
outputs/cifar10_eval/
```

### GTSRB

```bash
python GTSRB/feature-extraction.py
python SPLIT-DATA/split-gtsrb.py
python GTSRB/double-state-clustering.py
python GTSRB/SNN-training-gtsrb.py
python GTSRB/Pseudo-labeling-gtsrb.py
python GTSRB/moe-training-gtsrb.py
python GTSRB/Soft-routing-inference-gtsrb.py
```

Expected main artifact directories:

```text
outputs/gtsrb_features/
outputs/gtsrb_split/
outputs/gtsrb_clusters/
outputs/gtsrb_siamese/
outputs/gtsrb_pseudo/
outputs/gtsrb_moe/
outputs/gtsrb_eval/
```

GTSRB is substantially more expensive because it creates up to 43 experts and
runs separate Optuna studies for eligible experts.

## Zero-shot experiments

Zero-shot training requires extracted features and cluster IDs, so complete at
least feature extraction and clustering first. The intended commands are:

```bash
# MNIST
python MNIST/zero-shot-training-mnist.py
python MNIST/zero-shot-evaluation-mnist.py

# CIFAR-10
python CIFAR-10/zero-shot-training-cifar10.py
python CIFAR-10/zero-shot-evaluation-cifar10.py

# GTSRB
python GTSRB/zero-shot-training-gtsrb.py
python GTSRB/zero-shot-evaluation-gtsrb.py
```

These pairs do **not** currently run end-to-end without code corrections. The
trainers do not save `gating.pt`, while the evaluators require it from
`outputs/<dataset>_zeroshot/gating.pt`. CIFAR-10 training also incorrectly sets
`NUM_CLUSTERS = 43`. See the corrections section before using this branch.

The reported zero-shot accuracy is not label-free evaluation: true labels are
used after training to associate clusters with semantic classes and compute
accuracy/macro-F1.

## OOD-aware routing

OOD training requires a completed ordinary MoE run with
`outputs/<dataset>_moe/gating.pt`. It writes `gating_ood.pt` into the same MoE
directory. Run training before evaluation:

```bash
# MNIST versus Fashion-MNIST
python MNIST/OOD-training-mnist.py
python MNIST/OOD-evaluation-mnist.py

# CIFAR-10 versus SVHN
python CIFAR-10/OOD-training-cifar10.py
python CIFAR-10/OOD-evaluation-cifar10.py

# GTSRB versus SVHN
python GTSRB/OOD-training-gtsrb.py
python GTSRB/OOD-evaluation-gtsrb.py
```

The OOD gate objective combines:

1. cross-entropy routing loss on in-distribution cluster assignments; and
2. KL divergence toward a uniform expert distribution on OOD inputs.

At evaluation time, higher gate entropy is treated as evidence of an OOD input.
The scripts report ID accuracy, macro-F1, mean ID/OOD entropy, and AUROC.

## Output artifacts

For `<name>` equal to `mnist`, `cifar10`, or `gtsrb`, the main output contract is:

```text
outputs/
├── <name>_features/
│   ├── features.npy
│   ├── labels.npy
│   └── indices.npy
├── <name>_split/
│   ├── labeled_idx.npy
│   └── unlabeled_idx.npy
├── <name>_clusters/
│   └── cluster_ids.npy
├── <name>_siamese/
│   ├── siamese_encoder.pt
│   └── tau.npy
├── <name>_pseudo/
│   ├── pseudo_idx.npy
│   └── pseudo_labels.npy
├── <name>_moe/
│   ├── expert_0.pt
│   ├── expert_0_params.json
│   ├── ...
│   ├── gating.pt
│   └── gating_ood.pt               # after OOD training
├── <name>_eval/
│   └── confusion_matrix.npy
└── <name>_zeroshot/
    ├── gating.pt                    # intended, not currently saved
    └── confusion_matrix.npy
```

Most metrics are printed to standard output rather than persisted. Redirect logs
when reproducible records are needed:

```bash
mkdir -p logs
python MNIST/moe-training-mnist.py 2>&1 | tee logs/mnist_moe_training.log
```



## Reproducibility

For every experiment, record:

- dataset and exact torchvision version;
- official split policy and any correction applied to train/test handling;
- random seeds;
- labeled fraction and cluster count;
- feature-backbone weights and preprocessing;
- Siamese threshold and purity threshold;
- accepted pseudo-label count and class distribution;
- Optuna sampler, seed, number of trials, and best parameters;
- number of active/skipped experts;
- training epochs, batch sizes, learning rates, and weight decay;
- PyTorch, CUDA, GPU, and driver versions;
- OOD dataset, preprocessing, loss weight, and evaluation subset;
- source-code commit hash.

Generated `.npy` features can consume substantial disk space, while GTSRB's
expert checkpoints and Optuna searches can be expensive. Keep outputs from
different configurations in separately named directories rather than
overwriting a previous run.

## Publishing to GitHub

Commit source code and documentation, but normally exclude datasets, model
weights, extracted features, caches, logs, and environments. A suitable
`.gitignore` should include:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
data/
outputs/
logs/
*.pt
*.pth
*.npy
```

If trained weights or selected small artifacts must be published, use Git LFS or
a versioned release/artifact service. Document which source commit and
configuration produced every shared artifact.



