# Proposed Color-Invariant Metric Learning System Report

## Executive Summary
This report presents the performance of the **Proposed Color-Invariant Metric Learning Model** (`ColorInvariantResNet50`). Trained with **Supervised Contrastive Loss (SupCon)**, a custom **DesignBatchSampler**, and **Color-Invariant Data Augmentations**, the proposed model learns deep representations invariant to textile colorways.

---

## 1. Architecture & Design Specifications
- **Backbone**: ResNet-50 (Fine-tuned from `layer4`).
- **Projection Head**: `Linear(2048 -> 512) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(512 -> 256)`.
- **Embedding Dimension**: `256` (L2 Normalized).
- **Loss Function**: Supervised Contrastive Loss (SupCon, $\tau = 0.07$).
- **Sampler**: Design-Aware Batch Sampler (`DesignBatchSampler`).

---

## 2. Sampler Statistics
- **Total Training Designs**: `60`
- **Multi-Image Designs**: `16`
- **Singleton Designs**: `44`
- **Average Positive Pairs per Batch**: `3.57`
- **Batches with $\ge 1$ Positive Pair**: `100.0%`

---

## 3. Comparison: Proposed Model vs. Frozen ResNet-50 Baseline

### Task 1: Identification / Retrieval Performance

| Metric | Primary Baseline | Primary Proposed | Primary Delta | Secondary Baseline | Secondary Proposed | Secondary Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Recall@1** | 0.7500 | **0.8750** | **+0.1250** | 0.7143 | **0.8214** | **+0.1071** |
| **Recall@5** | 1.0000 | **1.0000** | **0.0000** | 1.0000 | **1.0000** | **0.0000** |
| **Recall@10** | 1.0000 | **1.0000** | **0.0000** | 1.0000 | **1.0000** | **0.0000** |
| **mAP** | 0.8581 | **0.8311** | **-0.0271** | 0.8166 | **0.8235** | **+0.0069** |

### Task 2: Pairwise Verification Performance

| Metric | Primary Baseline | Primary Proposed | Secondary Baseline | Secondary Proposed |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.9923 | **0.9696** | 0.9886 | **0.9740** |
| **F1-Score** | 0.7586 | **0.4259** | 0.7783 | **0.5461** |
| **Accuracy** | 0.9889 | **0.9506** | 0.9883 | **0.9666** |
| **Precision** | 0.7097 | **0.2840** | 0.7182 | **0.4074** |
| **Recall** | 0.8148 | **0.8519** | 0.8495 | **0.8280** |
