# DeepLure Baseline System Report: Pretrained ResNet-50 Saree Retrieval

## Executive Summary
This document establishes the zero-shot baseline performance for color-invariant saree design recognition using an un-finetuned, ImageNet-pretrained **ResNet-50** architecture. 

---

## 1. Architecture & Pipeline
- **Backbone**: Pretrained `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`.
- **Classification Layer**: Removed final FC layer (`Linear(2048 -> 1000)`).
- **Embedding Output**: 2048-dimensional continuous feature vector.
- **Normalisation**: L2-normalization ($||v||_2 = 1$).
- **Similarity Metric**: Cosine Similarity / Vector Dot Product.
- **Fine-Tuning**: **None** (Weights frozen; standard zero-shot feature extraction).

---

## 2. Preprocessing Specification
- **Resize**: Shortest side scaled to 256 pixels.
- **Crop**: Center crop to $224 \times 224$ pixels.
- **Tensor Conversion**: Scaled to $[0.0, 1.0]$ float tensor.
- **Standardization**: Mean $[0.485, 0.456, 0.406]$, Std $[0.229, 0.224, 0.225]$.

---

## 3. Quantitative Results

### Task 1: Identification / Retrieval Performance

| Metric | Primary Protocol (8 High-Conf Test Designs) | Secondary Protocol (Full 28 Multi-Image Groups) |
| :--- | :---: | :---: |
| **Recall@1** | **0.7500** | **0.7143** |
| **Recall@5** | **1.0000** | **1.0000** |
| **Recall@10** | **1.0000** | **1.0000** |
| **mAP** | **0.8581** | **0.8166** |

### Task 2: Pairwise Verification Performance

| Metric | Primary Protocol | Secondary Protocol |
| :--- | :---: | :---: |
| **ROC-AUC** | **0.9923** | **0.9886** |
| **F1-Score** | **0.7586** | **0.7783** |
| **Accuracy** | **0.9889** | **0.9883** |
| **Precision** | **0.7097** | **0.7182** |
| **Recall** | **0.8148** | **0.8495** |
| **Operating Threshold $\tau$** | **0.7100** | **0.7100** |

---

## 4. Key Limitations of Pretrained ResNet-50 Baseline

1. **Color Bias in ImageNet Embeddings**: Standard ResNet-50 features strongly encode dominant RGB color channels and background lighting. Consequently, sarees of completely different designs with similar color palettes are ranked higher than the same design in a different colorway.
2. **Lack of Motif Granularity**: Deep convolution layers aggregate global spatial statistics, diluting fine-grained geometric weave motifs, borders, and pallu patterns.
3. **Low Recall@1 Baseline**: The baseline achieves a Recall@1 of **75.0%**, demonstrating the absolute necessity for specialized color-invariant metric learning.
