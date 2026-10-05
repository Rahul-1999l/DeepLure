import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

# 1. DATA LEAKAGE & OVERLAP CHECK
pri_train = pd.read_csv(root_dir / "data/splits/primary_train.csv")
pri_val = pd.read_csv(root_dir / "data/splits/primary_val.csv")
pri_query = pd.read_csv(root_dir / "data/splits/primary_query.csv")
pri_gallery = pd.read_csv(root_dir / "data/splits/primary_gallery.csv")

sec_query = pd.read_csv(root_dir / "data/splits/secondary_query.csv")
sec_gallery = pd.read_csv(root_dir / "data/splits/secondary_gallery.csv")

train_paths = set(pri_train['image_path'])
val_paths = set(pri_val['image_path'])
query_paths = set(pri_query['image_path'])
gallery_paths = set(pri_gallery['image_path'])

train_val_overlap = train_paths.intersection(val_paths)
train_query_overlap = train_paths.intersection(query_paths)
query_in_gallery_overlap = query_paths.intersection(gallery_paths)

print("=== 1. LEAKAGE CHECKS ===")
print(f"Train / Val path overlap: {len(train_val_overlap)}")
print(f"Train / Query path overlap: {len(train_query_overlap)}")
print(f"Query / Gallery path overlap: {len(query_in_gallery_overlap)}")

train_dids = set(pri_train['design_id'])
val_dids = set(pri_val['design_id'])
query_dids = set(pri_query['design_id'])

print(f"Train / Val design_id overlap: {len(train_dids.intersection(val_dids))}")
print(f"Train / Query design_id overlap: {len(train_dids.intersection(query_dids))}")
print(f"Val / Query design_id overlap: {len(val_dids.intersection(query_dids))}")

# 2. MANUAL QUERY TRACE FOR RETRIEVAL (Query #1)
from src.models.baseline import ResNet50FeatureExtractor
from src.evaluation.retrieval import extract_dataset_embeddings

model = ResNet50FeatureExtractor().to('cpu')
model.eval()

q1_row = pri_query.iloc[0]
q1_did = q1_row['design_id']
q1_path = q1_row['image_path']

q_embeds = extract_dataset_embeddings(pri_query.iloc[:1], model, root_dir, device='cpu')
g_embeds = extract_dataset_embeddings(pri_gallery, model, root_dir, device='cpu')

sims = np.dot(q_embeds, g_embeds.T)[0]
ranked_idx = np.argsort(-sims)

print(f"\n=== 2. MANUAL RETRIEVAL TRACE FOR QUERY 1 ===")
print(f"Query Filename: {q1_path.split('/')[-1]}")
print(f"Query Path: {q1_path}")
print(f"Query Design ID: {q1_did}")
print("\nTop 10 Retrieved Gallery Items:")
for rank, idx in enumerate(ranked_idx[:10], 1):
    g_row = pri_gallery.iloc[idx]
    g_fn = g_row['image_path'].split('/')[-1]
    g_did = g_row['design_id']
    score = sims[idx]
    status = "SAME DESIGN" if g_did == q1_did else "DIFFERENT DESIGN"
    print(f"  Rank {rank:02d} | Similarity: {score:.4f} | Filename: {g_fn} | Design ID: {g_did} | Status: {status}")

# 3. VERIFY NUMERICAL REPRODUCIBILITY OF SAVED RESULTS
with open(root_dir / "outputs/baseline/baseline_results.json") as f:
    base_json = json.load(f)

with open(root_dir / "outputs/ablation/ablation_results.json") as f:
    abl_json = json.load(f)

print("\n=== 3. VERIFY SAVED NUMBERS IN JSON FILES ===")
print("Baseline Primary R@1:", base_json['primary_protocol_metrics']['retrieval']['recall_at_1'])
print("Baseline Primary mAP:", base_json['primary_protocol_metrics']['retrieval']['mAP'])
print("Baseline Primary ROC-AUC:", base_json['primary_protocol_metrics']['verification']['roc_auc'])

for exp in abl_json['experiments']:
    print(f"{exp['name']}: Primary R@1={exp['primary_r1']:.4f}, Primary mAP={exp['primary_mAP']:.4f}, Secondary R@1={exp['secondary_r1']:.4f}, Secondary mAP={exp['secondary_mAP']:.4f}")
