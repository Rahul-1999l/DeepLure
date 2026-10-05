# DeepLure — Color-Invariant Saree Design Retrieval

DeepLure is a computer-vision project for recognizing and retrieving the **same saree design across different colorways**. It compares a frozen ImageNet-pretrained ResNet-50 baseline with a metric-learning model trained using **Supervised Contrastive Loss (SupCon)**, a design-aware sampler, and color-invariant augmentations.

## Highlights

- **165 images** covering **72 saree designs**
- ResNet-50 baseline with 2048-D normalized features
- Proposed `ColorInvariantResNet50` with a **256-D L2-normalized embedding**
- Supervised Contrastive Loss with design-aware batches
- ColorJitter and RandomGrayscale augmentation to reduce color dependence
- Retrieval metrics: Recall@1, Recall@5, Recall@10 and mAP
- Pairwise verification metrics: ROC-AUC, accuracy, precision, recall and F1
- Leakage-aware train/validation/query/gallery splits
- Ablation studies and qualitative top-5 retrieval visualizations

## Results

| Model | Primary R@1 | Primary mAP | Secondary R@1 | Secondary mAP |
|---|---:|---:|---:|---:|
| Frozen ResNet-50 baseline | 0.7500 | 0.8581 | 0.7143 | 0.8166 |
| Proposed color-invariant model | **0.8750** | 0.8311 | **0.8214** | **0.8235** |

The proposed model improves Recall@1 by **12.5 percentage points** on the primary protocol and **10.7 percentage points** on the secondary protocol. See `outputs/` for the complete reports, metrics, ablations and retrieval examples.

> Note: this is a small experimental dataset, so the reported metrics should be interpreted as project results rather than production-scale benchmarks.

## Architecture

The proposed pipeline uses:

1. An ImageNet-pretrained **ResNet-50** backbone.
2. Fine-tuning from `layer4`.
3. A projection head: `2048 → 512 → 256`.
4. L2-normalized embeddings.
5. **Supervised Contrastive Loss** (`temperature = 0.07`).
6. A custom `DesignBatchSampler` to place positive design pairs in training batches.
7. Color-oriented augmentation to encourage motif/texture learning instead of reliance on dye color.

## Repository Structure

```text
DeepLure/
├── data/
│   ├── metadata.csv
│   └── splits/
├── outputs/
│   ├── ablation/
│   ├── audit/
│   ├── baseline/
│   ├── dataset_inspection/
│   ├── proposed/
│   └── evaluation_protocol.md
├── sarees_dataset/
├── scripts/
│   ├── audit_groups.py
│   ├── create_splits.py
│   ├── generate_metadata.py
│   ├── run_ablations.py
│   ├── run_baseline.py
│   └── train_proposed.py
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── losses/
│   ├── models/
│   └── training/
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Python 3.10+ is recommended.

```bash
git clone <your-repository-url>
cd DeepLure

python -m venv .venv
```

Activate the environment:

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

The scripts now resolve the repository path automatically, so no machine-specific absolute path is required.

If starting from the included dataset, regenerate metadata and splits with:

```bash
python scripts/generate_metadata.py
python scripts/audit_groups.py
python scripts/create_splits.py
```

Run the frozen ResNet-50 baseline:

```bash
python scripts/run_baseline.py
```

Train and evaluate the proposed model:

```bash
python scripts/train_proposed.py
```

Run the ablation experiments:

```bash
python scripts/run_ablations.py
```

The first run may download ImageNet-pretrained ResNet-50 weights through `torchvision`.

## Evaluation Protocol

DeepLure treats two images as a positive pair when they share the same `design_id` while representing distinct images/colorways. Query and gallery roles are separated for retrieval evaluation, and the project includes explicit split definitions under `data/splits/`.

For details, see [`outputs/evaluation_protocol.md`](outputs/evaluation_protocol.md).

## Outputs

The repository keeps lightweight experiment artifacts such as:

- JSON metric files
- Markdown reports
- Dataset inspection summaries
- Retrieval result images
- Ablation comparison plots

Large trained checkpoint files (`*.pth`, `*.pt`, `*.ckpt`) are intentionally excluded from Git because they are generated artifacts and can exceed normal repository file-size limits.

## Tech Stack

Python · PyTorch · Torchvision · NumPy · Pandas · scikit-learn · OpenCV · Pillow · Matplotlib · NetworkX

## Suggested GitHub Description

> Deep learning-based saree design retrieval and verification using ResNet-50, supervised contrastive learning, and color-invariant feature embeddings.
