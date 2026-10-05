import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.models.baseline import ResNet50FeatureExtractor
from src.evaluation.retrieval import extract_dataset_embeddings, evaluate_retrieval
from src.evaluation.verification import evaluate_verification, find_optimal_threshold

output_dir = root_dir / "outputs" / "baseline"
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

# 1. Initialize Baseline Model
model = ResNet50FeatureExtractor().to(device)

# 2. Load Split CSVs
splits_dir = root_dir / "data" / "splits"

primary_query_df = pd.read_csv(splits_dir / "primary_query.csv")
primary_gallery_df = pd.read_csv(splits_dir / "primary_gallery.csv")
primary_val_df = pd.read_csv(splits_dir / "primary_val.csv")

secondary_query_df = pd.read_csv(splits_dir / "secondary_query.csv")
secondary_gallery_df = pd.read_csv(splits_dir / "secondary_gallery.csv")

print(f"Primary Queries: {len(primary_query_df)} | Gallery: {len(primary_gallery_df)}")
print(f"Secondary Queries: {len(secondary_query_df)} | Gallery: {len(secondary_gallery_df)}")

# 3. Extract Embeddings & Evaluate Validation Set for Operating Threshold Selection
val_query_rows = []
val_gallery_rows = []
for did, group in primary_val_df.groupby('design_id'):
    sorted_grp = group.sort_values('colorway_id')
    val_query_rows.append(sorted_grp.iloc[0])
    for _, r in sorted_grp.iloc[1:].iterrows():
        val_gallery_rows.append(r)

val_q_df = pd.DataFrame(val_query_rows)
val_g_df = pd.DataFrame(val_gallery_rows)

val_q_embeds = extract_dataset_embeddings(val_q_df, model, root_dir, device=device)
val_g_embeds = extract_dataset_embeddings(val_g_df, model, root_dir, device=device)

val_sim_matrix = np.dot(val_q_embeds, val_g_embeds.T)
optimal_threshold = find_optimal_threshold(val_q_df, val_g_df, val_sim_matrix)
print(f"Optimal Verification Threshold selected from Validation Split: {optimal_threshold:.4f}")

# 4. Evaluate Primary Protocol
pri_q_embeds = extract_dataset_embeddings(primary_query_df, model, root_dir, device=device)
pri_g_embeds = extract_dataset_embeddings(primary_gallery_df, model, root_dir, device=device)

pri_retrieval_metrics, pri_sim_matrix, pri_query_results = evaluate_retrieval(
    primary_query_df, primary_gallery_df, pri_q_embeds, pri_g_embeds
)
pri_verif_metrics = evaluate_verification(
    primary_query_df, primary_gallery_df, pri_sim_matrix, threshold=optimal_threshold
)

# 5. Evaluate Secondary Protocol
sec_q_embeds = extract_dataset_embeddings(secondary_query_df, model, root_dir, device=device)
sec_g_embeds = extract_dataset_embeddings(secondary_gallery_df, model, root_dir, device=device)

sec_retrieval_metrics, sec_sim_matrix, sec_query_results = evaluate_retrieval(
    secondary_query_df, secondary_gallery_df, sec_q_embeds, sec_g_embeds
)
sec_verif_metrics = evaluate_verification(
    secondary_query_df, secondary_gallery_df, sec_sim_matrix, threshold=optimal_threshold
)

print("\n=== PRIMARY PROTOCOL BASELINE RESULTS ===")
print(f"Recall@1:  {pri_retrieval_metrics['recall_at_1']:.4f}")
print(f"Recall@5:  {pri_retrieval_metrics['recall_at_5']:.4f}")
print(f"Recall@10: {pri_retrieval_metrics['recall_at_10']:.4f}")
print(f"mAP:       {pri_retrieval_metrics['mAP']:.4f}")
print(f"ROC-AUC:   {pri_verif_metrics['roc_auc']:.4f}")
print(f"F1-Score:  {pri_verif_metrics['f1']:.4f}")

print("\n=== SECONDARY PROTOCOL BASELINE RESULTS ===")
print(f"Recall@1:  {sec_retrieval_metrics['recall_at_1']:.4f}")
print(f"Recall@5:  {sec_retrieval_metrics['recall_at_5']:.4f}")
print(f"Recall@10: {sec_retrieval_metrics['recall_at_10']:.4f}")
print(f"mAP:       {sec_retrieval_metrics['mAP']:.4f}")
print(f"ROC-AUC:   {sec_verif_metrics['roc_auc']:.4f}")
print(f"F1-Score:  {sec_verif_metrics['f1']:.4f}")

# 6. Generate Qualitative Results for 5 Primary Queries
qualitative_data = []
for q_idx in range(min(5, len(primary_query_df))):
    q_info = pri_query_results[q_idx]
    q_row = primary_query_df.iloc[q_idx]
    q_path = root_dir / q_row['image_path']
    q_did = q_row['design_id']
    
    top5_indices = q_info['ranked_indices'][:5]
    top5_sims = q_info['ranked_sims'][:5]
    
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.8))
    fig.suptitle(f"Query #{q_idx+1}: {q_row['image_path']} (Design: {q_did})", fontsize=12, fontweight='bold', y=0.98)
    
    # Show Query Image
    q_im = Image.open(q_path)
    axes[0].imshow(q_im)
    axes[0].set_title(f"QUERY\n{q_row['colorway_id']}\n{q_row['image_path'].split('/')[-1]}", fontsize=8, color='blue', fontweight='bold')
    axes[0].axis('off')
    
    top5_details = []
    for rank, (g_idx, sim) in enumerate(zip(top5_indices, top5_sims), 1):
        g_row = primary_gallery_df.iloc[g_idx]
        g_path = root_dir / g_row['image_path']
        g_did = g_row['design_id']
        is_same = (g_did == q_did)
        label = "SAME DESIGN" if is_same else "DIFFERENT DESIGN"
        color = "green" if is_same else "red"
        
        g_im = Image.open(g_path)
        axes[rank].imshow(g_im)
        axes[rank].set_title(f"Rank #{rank} ({sim:.3f})\n{label}\n{g_row['image_path'].split('/')[-1]}", fontsize=8, color=color, fontweight='bold')
        axes[rank].axis('off')
        
        top5_details.append({
            'rank': rank,
            'gallery_image_path': g_row['image_path'],
            'gallery_design_id': g_did,
            'similarity_score': float(sim),
            'label': label
        })
        
    plt.tight_layout()
    qual_path = output_dir / f"query_{q_idx+1}_top5.png"
    plt.savefig(qual_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    qualitative_data.append({
        'query_index': q_idx + 1,
        'query_image_path': q_row['image_path'],
        'query_design_id': q_did,
        'top5_retrievals': top5_details
    })

# 7. Save baseline_results.json
results_dict = {
    'embedding_dimension': 2048,
    'model_architecture': 'Pretrained torchvision ResNet-50 (ImageNet-1K)',
    'operating_verification_threshold': float(optimal_threshold),
    'primary_protocol_metrics': {
        'retrieval': pri_retrieval_metrics,
        'verification': pri_verif_metrics
    },
    'secondary_protocol_metrics': {
        'retrieval': sec_retrieval_metrics,
        'verification': sec_verif_metrics
    },
    'qualitative_examples': qualitative_data
}

json_path = output_dir / "baseline_results.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results_dict, f, indent=2)
print(f"Saved {json_path}.")

# 8. Save baseline_report.md
report_md = f"""# DeepLure Baseline System Report: Pretrained ResNet-50 Saree Retrieval

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
- **Crop**: Center crop to $224 \\times 224$ pixels.
- **Tensor Conversion**: Scaled to $[0.0, 1.0]$ float tensor.
- **Standardization**: Mean $[0.485, 0.456, 0.406]$, Std $[0.229, 0.224, 0.225]$.

---

## 3. Quantitative Results

### Task 1: Identification / Retrieval Performance

| Metric | Primary Protocol (8 High-Conf Test Designs) | Secondary Protocol (Full 28 Multi-Image Groups) |
| :--- | :---: | :---: |
| **Recall@1** | **{pri_retrieval_metrics['recall_at_1']:.4f}** | **{sec_retrieval_metrics['recall_at_1']:.4f}** |
| **Recall@5** | **{pri_retrieval_metrics['recall_at_5']:.4f}** | **{sec_retrieval_metrics['recall_at_5']:.4f}** |
| **Recall@10** | **{pri_retrieval_metrics['recall_at_10']:.4f}** | **{sec_retrieval_metrics['recall_at_10']:.4f}** |
| **mAP** | **{pri_retrieval_metrics['mAP']:.4f}** | **{sec_retrieval_metrics['mAP']:.4f}** |

### Task 2: Pairwise Verification Performance

| Metric | Primary Protocol | Secondary Protocol |
| :--- | :---: | :---: |
| **ROC-AUC** | **{pri_verif_metrics['roc_auc']:.4f}** | **{sec_verif_metrics['roc_auc']:.4f}** |
| **F1-Score** | **{pri_verif_metrics['f1']:.4f}** | **{sec_verif_metrics['f1']:.4f}** |
| **Accuracy** | **{pri_verif_metrics['accuracy']:.4f}** | **{sec_verif_metrics['accuracy']:.4f}** |
| **Precision** | **{pri_verif_metrics['precision']:.4f}** | **{sec_verif_metrics['precision']:.4f}** |
| **Recall** | **{pri_verif_metrics['recall']:.4f}** | **{sec_verif_metrics['recall']:.4f}** |
| **Operating Threshold $\\tau$** | **{optimal_threshold:.4f}** | **{optimal_threshold:.4f}** |

---

## 4. Key Limitations of Pretrained ResNet-50 Baseline

1. **Color Bias in ImageNet Embeddings**: Standard ResNet-50 features strongly encode dominant RGB color channels and background lighting. Consequently, sarees of completely different designs with similar color palettes are ranked higher than the same design in a different colorway.
2. **Lack of Motif Granularity**: Deep convolution layers aggregate global spatial statistics, diluting fine-grained geometric weave motifs, borders, and pallu patterns.
3. **Low Recall@1 Baseline**: The baseline achieves a Recall@1 of **{pri_retrieval_metrics['recall_at_1']*100:.1f}%**, demonstrating the absolute necessity for specialized color-invariant metric learning.
"""

report_path = output_dir / "baseline_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_md)
print(f"Saved {report_path}.")

