# CRISMER

**CRISMER**: A transformer-based Interpretable Deep Learning Approach for Genome-wide CRISPR Cas-9 Off-Target Prediction and Optimization.

---

## Overview

CRISMER is a hybrid deep learning framework designed to predict CRISPR-Cas9 off-target cleavage activity and optimize single-guide RNA (sgRNA) design. It incorporates:
*   **Multi-Branch CNN with CBAM**: Extracts local *k-mer* features using spatial and channel attention mechanisms (Convolutional Block Attention Module).
*   **Transformer Encoder**: Captures long-range sequence dependencies across guide-target alignments.
*   **Explainable AI (XAI)**: Supports sequence interpretability to highlight mismatch locations and PAM-proximal regions.

---

## Hardware Environment

All model training, evaluation, and ablation experiments in this work were conducted on **Kaggle** utilizing a single **NVIDIA Tesla P100 GPU** (16GB VRAM) and PyTorch.

---

## Repository Structure

```
├── Ablation Studies/      # Notebooks exploring CNN branch configurations, attention mechanisms, and layer exclusions
├── datasets/              # Datasets used for model training, testing, and independent validation
├── models/                # Saved pre-trained model checkpoints
├── crismer-opti/          # CLI tool (CRISMER.py) for scoring, specificity estimation, and sgRNA optimization
├── test-scenario-1.ipynb  # Scenario 1: Mixed off-target prediction
├── test-scenario-2.ipynb  # Scenario 2: Model generalization on independent datasets
└── test-scenario-3.ipynb  # Scenario 3: Off-target prediction with bulges (indels)
```

### Experimental Scenarios

1.  **Scenario 1 (`test-scenario-1.ipynb`)**: Evaluates performance using a stratified 80/20 train-test split on the mixed off-target dataset `datasets/all_off_target.csv`.
2.  **Scenario 2 (`test-scenario-2.ipynb`)**: Tests cross-dataset generalization. The model is trained on `datasets/changeseq_siteseq.csv` and evaluated against four independent datasets: `circleseq_all.csv`, `guideseq.csv`, `surroseq.csv`, and `ttiss.csv`.
3.  **Scenario 3 (`test-scenario-3.ipynb`)**: Addresses off-target prediction with insertion/deletion bulges using `datasets/I1.txt` and a custom 24-dimensional alignment encoding scheme.

---

## Guide RNA Design & Optimization (CRISMER-opti)

For installing dependencies, using the command-line utility for specificity calculations, activity scoring, and guide RNA optimization via **Cas-OFFinder**, please refer to the detailed instructions in [crismer-opti/README.md](file:///d:/CRISMER/crismer-opti/README.md).

---

## Citation

*Citation details will be added here upon publication.*
