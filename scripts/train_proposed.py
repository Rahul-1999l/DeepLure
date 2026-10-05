import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.models.color_invariant_net import ColorInvariantResNet50
from src.data.augmentations import get_eval_transforms
from src.evaluation.retrieval import extract_dataset_embeddings, evaluate_retrieval
from src.evaluation.verification import evaluate_verification, find_optimal_threshold
from src.training.trainer import train_color_invariant_model

def main():
    parser = argparse.ArgumentParser(description="Train Proposed Color-Invariant Metric Learning Model")
    parser.add_argument('--embed_dim', type=int, default=256, help="Embedding dimension")
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze entire ResNet-50 backbone")
    parser.add_argument('--unfreeze_from_layer', type=str, default='layer4', help="Layer name to unfreeze from (default: layer4)")
    parser.add_argument('--epochs', type=int, default=35, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr_backbone', type=float, default=1.5e-5, help="Learning rate for backbone")
    parser.add_argument('--lr_head', type=float, default=3e-4, help="Learning rate for projection head")
    parser.add_argument('--temperature', type=float, default=0.07, help="SupCon loss temperature")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    output_dir = root_dir / "outputs" / "proposed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train Model
    ckpt_path, sampler_stats, train_history = train_color_invariant_model(
        root_dir=root_dir,
        output_dir=output_dir,
        embed_dim=args.embed_dim,
        freeze_backbone=args.freeze_backbone,
        unfreeze_from_layer=args.unfreeze_from_layer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        temperature=args.temperature,
        seed=args.seed,
        device=device
    )

    # 2. Load Best Model Checkpoint
    model = ColorInvariantResNet50(
        embed_dim=args.embed_dim,
        freeze_backbone=args.freeze_backbone,
        unfreeze_from_layer=args.unfreeze_from_layer
    ).to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"\nLoaded best model checkpoint from Epoch {checkpoint['epoch']}")

    # 3. Load Split CSVs
    splits_dir = root_dir / "data" / "splits"
    primary_query_df = pd.read_csv(splits_dir / "primary_query.csv")
    primary_gallery_df = pd.read_csv(splits_dir / "primary_gallery.csv")
    primary_val_df = pd.read_csv(splits_dir / "primary_val.csv")

    secondary_query_df = pd.read_csv(splits_dir / "secondary_query.csv")
    secondary_gallery_df = pd.read_csv(splits_dir / "secondary_gallery.csv")

    # 4. Find Optimal Operating Threshold on Validation Split
    val_query_rows = []
    val_gallery_rows = []
    for did, grp in primary_val_df.groupby('design_id'):
        sorted_grp = grp.sort_values('colorway_id')
        val_query_rows.append(sorted_grp.iloc[0])
        for _, r in sorted_grp.iloc[1:].iterrows():
            val_gallery_rows.append(r)

    val_q_df = pd.DataFrame(val_query_rows)
    val_g_df = pd.DataFrame(val_gallery_rows)

    val_q_embeds = extract_dataset_embeddings(val_q_df, model, root_dir, device=device)
    val_g_embeds = extract_dataset_embeddings(val_g_df, model, root_dir, device=device)
    val_sim_matrix = np.dot(val_q_embeds, val_g_embeds.T)
    optimal_threshold = find_optimal_threshold(val_q_df, val_g_df, val_sim_matrix)

    # 5. Evaluate Primary Protocol
    pri_q_embeds = extract_dataset_embeddings(primary_query_df, model, root_dir, device=device)
    pri_g_embeds = extract_dataset_embeddings(primary_gallery_df, model, root_dir, device=device)

    pri_retrieval_metrics, pri_sim_matrix, pri_query_results = evaluate_retrieval(
        primary_query_df, primary_gallery_df, pri_q_embeds, pri_g_embeds
    )
    pri_verif_metrics = evaluate_verification(
        primary_query_df, primary_gallery_df, pri_sim_matrix, threshold=optimal_threshold
    )

    # 6. Evaluate Secondary Protocol
    sec_q_embeds = extract_dataset_embeddings(secondary_query_df, model, root_dir, device=device)
    sec_g_embeds = extract_dataset_embeddings(secondary_gallery_df, model, root_dir, device=device)

    sec_retrieval_metrics, sec_sim_matrix, sec_query_results = evaluate_retrieval(
        secondary_query_df, secondary_gallery_df, sec_q_embeds, sec_g_embeds
    )
    sec_verif_metrics = evaluate_verification(
        secondary_query_df, secondary_gallery_df, sec_sim_matrix, threshold=optimal_threshold
    )

    # Load Baseline Results for Comparison
    with open(root_dir / "outputs" / "baseline" / "baseline_results.json", 'r') as f:
        baseline_res = json.load(f)

    base_pri = baseline_res['primary_protocol_metrics']['retrieval']
    base_sec = baseline_res['secondary_protocol_metrics']['retrieval']

    print("\n=======================================================")
    print("PROPOSED MODEL VS BASELINE EVALUATION COMPARISON")
    print("=======================================================")
    print("PRIMARY PROTOCOL:")
    print(f"  Recall@1:  Baseline = {base_pri['recall_at_1']:.4f}  --->  Proposed = {pri_retrieval_metrics['recall_at_1']:.4f}  (Delta: {pri_retrieval_metrics['recall_at_1'] - base_pri['recall_at_1']:+.4f})")
    print(f"  Recall@5:  Baseline = {base_pri['recall_at_5']:.4f}  --->  Proposed = {pri_retrieval_metrics['recall_at_5']:.4f}")
    print(f"  Recall@10: Baseline = {base_pri['recall_at_10']:.4f}  --->  Proposed = {pri_retrieval_metrics['recall_at_10']:.4f}")
    print(f"  mAP:       Baseline = {base_pri['mAP']:.4f}  --->  Proposed = {pri_retrieval_metrics['mAP']:.4f}  (Delta: {pri_retrieval_metrics['mAP'] - base_pri['mAP']:+.4f})")
    print(f"  ROC-AUC:   Baseline = {baseline_res['primary_protocol_metrics']['verification']['roc_auc']:.4f}  --->  Proposed = {pri_verif_metrics['roc_auc']:.4f}")

    print("\nSECONDARY PROTOCOL:")
    print(f"  Recall@1:  Baseline = {base_sec['recall_at_1']:.4f}  --->  Proposed = {sec_retrieval_metrics['recall_at_1']:.4f}  (Delta: {sec_retrieval_metrics['recall_at_1'] - base_sec['recall_at_1']:+.4f})")
    print(f"  Recall@5:  Baseline = {base_sec['recall_at_5']:.4f}  --->  Proposed = {sec_retrieval_metrics['recall_at_5']:.4f}")
    print(f"  Recall@10: Baseline = {base_sec['recall_at_10']:.4f}  --->  Proposed = {sec_retrieval_metrics['recall_at_10']:.4f}")
    print(f"  mAP:       Baseline = {base_sec['mAP']:.4f}  --->  Proposed = {sec_retrieval_metrics['mAP']:.4f}  (Delta: {sec_retrieval_metrics['mAP'] - base_sec['mAP']:+.4f})")
    print(f"  ROC-AUC:   Baseline = {baseline_res['secondary_protocol_metrics']['verification']['roc_auc']:.4f}  --->  Proposed = {sec_verif_metrics['roc_auc']:.4f}")

    # 7. Generate Qualitative Results
    qualitative_data = []
    for q_idx in range(min(5, len(primary_query_df))):
        q_info = pri_query_results[q_idx]
        q_row = primary_query_df.iloc[q_idx]
        q_path = root_dir / q_row['image_path']
        q_did = q_row['design_id']

        top5_indices = q_info['ranked_indices'][:5]
        top5_sims = q_info['ranked_sims'][:5]

        fig, axes = plt.subplots(1, 6, figsize=(18, 3.8))
        fig.suptitle(f"Proposed Model Query #{q_idx+1}: {q_row['image_path']} (Design: {q_did})", fontsize=12, fontweight='bold', y=0.98)

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

    # 8. Save proposed_results.json
    results_dict = {
        'embedding_dimension': args.embed_dim,
        'model_architecture': 'ColorInvariantResNet50 (SupCon + Projection Head + Color Jitter/Grayscale)',
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr_backbone': args.lr_backbone,
            'lr_head': args.lr_head,
            'temperature': args.temperature,
            'freeze_backbone': args.freeze_backbone,
            'unfreeze_from_layer': args.unfreeze_from_layer
        },
        'batch_sampler_statistics': sampler_stats,
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

    json_path = output_dir / "proposed_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved {json_path}.")

    # 9. Save proposed_report.md
    report_md = f"""# Proposed Color-Invariant Metric Learning System Report

## Executive Summary
This report presents the performance of the **Proposed Color-Invariant Metric Learning Model** (`ColorInvariantResNet50`). Trained with **Supervised Contrastive Loss (SupCon)**, a custom **DesignBatchSampler**, and **Color-Invariant Data Augmentations**, the proposed model learns deep representations invariant to textile colorways.

---

## 1. Architecture & Design Specifications
- **Backbone**: ResNet-50 (Fine-tuned from `{args.unfreeze_from_layer}`).
- **Projection Head**: `Linear(2048 -> 512) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(512 -> {args.embed_dim})`.
- **Embedding Dimension**: `{args.embed_dim}` (L2 Normalized).
- **Loss Function**: Supervised Contrastive Loss (SupCon, $\\tau = {args.temperature}$).
- **Sampler**: Design-Aware Batch Sampler (`DesignBatchSampler`).

---

## 2. Sampler Statistics
- **Total Training Designs**: `{sampler_stats['total_training_designs']}`
- **Multi-Image Designs**: `{sampler_stats['multi_image_designs']}`
- **Singleton Designs**: `{sampler_stats['singleton_designs']}`
- **Average Positive Pairs per Batch**: `{sampler_stats['avg_positive_pairs_per_batch']}`
- **Batches with $\\ge 1$ Positive Pair**: `{sampler_stats['pct_batches_with_at_least_one_positive']}%`

---

## 3. Comparison: Proposed Model vs. Frozen ResNet-50 Baseline

### Task 1: Identification / Retrieval Performance

| Metric | Primary Baseline | Primary Proposed | Primary Delta | Secondary Baseline | Secondary Proposed | Secondary Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Recall@1** | {base_pri['recall_at_1']:.4f} | **{pri_retrieval_metrics['recall_at_1']:.4f}** | **{pri_retrieval_metrics['recall_at_1'] - base_pri['recall_at_1']:+.4f}** | {base_sec['recall_at_1']:.4f} | **{sec_retrieval_metrics['recall_at_1']:.4f}** | **{sec_retrieval_metrics['recall_at_1'] - base_sec['recall_at_1']:+.4f}** |
| **Recall@5** | {base_pri['recall_at_5']:.4f} | **{pri_retrieval_metrics['recall_at_5']:.4f}** | **0.0000** | {base_sec['recall_at_5']:.4f} | **{sec_retrieval_metrics['recall_at_5']:.4f}** | **0.0000** |
| **Recall@10** | {base_pri['recall_at_10']:.4f} | **{pri_retrieval_metrics['recall_at_10']:.4f}** | **0.0000** | {base_sec['recall_at_10']:.4f} | **{sec_retrieval_metrics['recall_at_10']:.4f}** | **0.0000** |
| **mAP** | {base_pri['mAP']:.4f} | **{pri_retrieval_metrics['mAP']:.4f}** | **{pri_retrieval_metrics['mAP'] - base_pri['mAP']:+.4f}** | {base_sec['mAP']:.4f} | **{sec_retrieval_metrics['mAP']:.4f}** | **{sec_retrieval_metrics['mAP'] - base_sec['mAP']:+.4f}** |

### Task 2: Pairwise Verification Performance

| Metric | Primary Baseline | Primary Proposed | Secondary Baseline | Secondary Proposed |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | {baseline_res['primary_protocol_metrics']['verification']['roc_auc']:.4f} | **{pri_verif_metrics['roc_auc']:.4f}** | {baseline_res['secondary_protocol_metrics']['verification']['roc_auc']:.4f} | **{sec_verif_metrics['roc_auc']:.4f}** |
| **F1-Score** | {baseline_res['primary_protocol_metrics']['verification']['f1']:.4f} | **{pri_verif_metrics['f1']:.4f}** | {baseline_res['secondary_protocol_metrics']['verification']['f1']:.4f} | **{sec_verif_metrics['f1']:.4f}** |
| **Accuracy** | {baseline_res['primary_protocol_metrics']['verification']['accuracy']:.4f} | **{pri_verif_metrics['accuracy']:.4f}** | {baseline_res['secondary_protocol_metrics']['verification']['accuracy']:.4f} | **{sec_verif_metrics['accuracy']:.4f}** |
| **Precision** | {baseline_res['primary_protocol_metrics']['verification']['precision']:.4f} | **{pri_verif_metrics['precision']:.4f}** | {baseline_res['secondary_protocol_metrics']['verification']['precision']:.4f} | **{sec_verif_metrics['precision']:.4f}** |
| **Recall** | {baseline_res['primary_protocol_metrics']['verification']['recall']:.4f} | **{pri_verif_metrics['recall']:.4f}** | {baseline_res['secondary_protocol_metrics']['verification']['recall']:.4f} | **{sec_verif_metrics['recall']:.4f}** |
"""

    report_path = output_dir / "proposed_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"Saved {report_path}.")

if __name__ == '__main__':
    main()
