# DeepLure Evaluation Protocol: Color-Invariant Saree Design Recognition

## Executive Summary
This document establishes a leakage-safe evaluation protocol for the DeepLure assignment. The core task requirement dictates that **"the same saree design presented in a different color palette must be recognized as a match."**

---

## 1. Definition of Positive & Negative Pairs

### Task 1: Identification / Retrieval
- **Positive Pair**: A tuple $(Q_i, G_j)$ where Query image $Q_i$ and Gallery image $G_j$ belong to the **same `design_id`**, but $Q_i$ and $G_j$ represent **different colorways** ($Q_i \neq G_j$).
- **Negative Pair**: A tuple $(Q_i, G_k)$ where $Q_i$ and $G_k$ belong to **different `design_id`s**.

### Task 2: Verification (Pairwise Decision)
- **Positive Pair**: Any pair of images $(I_A, I_B)$ sharing the same `design_id` ($I_A \neq I_B$).
- **Negative Pair**: Any pair of images $(I_A, I_B)$ with different `design_id`s.

---

## 2. Prevention of Data Leakage

Data leakage is strictly prevented through **Design-Disjoint & Role-Disjoint Grouping**:
1. **Zero Intra-Design Split Across Train and Test**: No images belonging to the same `design_id` are ever divided between the training set and the test set.
2. **Strict Query-Gallery Role Separation**: For held-out test designs, exactly one colorway image per design is designated as the **Query**, while the remaining colorways are placed in the **Gallery**.
3. **No Query Embedding Leakage**: The gallery embeddings are computed independently of query embeddings.

---

## 3. Analysis of Evaluation Strategies

| Metric / Dimension | Strategy A: Standard Group Split | Strategy B: Colorway Holdout (Primary) | Strategy C: Disjoint Zero-Shot (Secondary) |
| :--- | :--- | :--- | :--- |
| **Description** | Random 60/20/20 GroupKFold on 28 Multi-Image Groups | 8 High-Confidence Held-Out Test Designs + 1 Colorway Query / Rest Gallery | 8 Completely Unseen Test Designs evaluated against 140 Gallery Distractors |
| **Train Images** | 73 images (16 designs + 44 singletons) | **107 images** (16 multi + 44 singletons) | **107 images** (16 multi + 44 singletons) |
| **Validation Images**| 24 images (6 designs) | **23 images** (4 multi designs) | **23 images** (4 multi designs) |
| **Test Query Images**| 5 images | **8 images** (1 per high-conf test design) | **8 images** (1 per unseen test design) |
| **Test Gallery Images**| 20 images + 44 singletons = 64 | **157 images** (27 test colorways + 130 distractors) | **140 images** (30 test colorways + 110 distractors) |
| **Positive Matches**| 19 matches | **27 positive matches** | **30 positive matches** |
| **Negative Pairings**| 320 pairings | **1,229 pairings** | **1,096 pairings** |
| **Designs Represented**| 28 multi-image designs | **72 total designs** | **72 total designs** |
| **Core Limitation**| Low query sample size in test split | Restricted to 8 high-confidence test designs for test phase | Requires zero-shot generalization to unseen weave patterns |

---

## 4. Recommended Evaluation Protocols

### Primary Protocol: Strategy B (High-Confidence Colorway Holdout)
- **Why Selected**: Directly tests the exact prompt requirement ("same design in a different palette must match").
- **Setup**:
  - **Training Set**: 107 images (16 multi-image designs + 44 singletons).
  - **Validation Set**: 23 images (4 multi-image designs).
  - **Query Set**: 8 query images (1 colorway per high-confidence test design).
  - **Gallery Set**: 157 gallery images (27 remaining test colorways + 130 background distractors).
  - **Valid Positive Matches**: 27.
  - **Negative Pairings**: 1,229.

### Secondary Protocol: Strategy B-Full (All-Design 28-Group Colorway Holdout)
- **Setup**: Evaluates across all 28 multi-image designs (28 queries vs 137 gallery items, 93 positive matches, 3,743 negative pairings).

---

## 5. Handling Singletons & Suspicious Groups

1. **Singleton Designs (44 images)**:
   - Cannot serve as positive retrieval queries because no second colorway exists in the dataset.
   - Placed in the **Training Set** and **Gallery Set** as negative background distractors to challenge the retrieval model.

2. **Suspicious Groups (20 groups)**:
   - Excluded from the Primary Test Query Set to ensure test ground-truth purity.
   - Retained in the Training Set to allow representation learning across varied saree motifs.

---

## 6. Official Reporting Metrics

### Task 1: Identification / Retrieval
- **Recall@1**: Proportion of queries where the top-1 retrieved gallery item is a positive match.
- **Recall@5**: Proportion of queries where at least 1 positive match is in top-5.
- **Recall@10**: Proportion of queries where at least 1 positive match is in top-10.
- **mAP (mean Average Precision)**: Area under precision-recall curve across all queries.

### Task 2: Verification
- **Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC** at optimal distance threshold $\tau$.
