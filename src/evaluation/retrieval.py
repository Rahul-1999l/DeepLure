import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from src.data.augmentations import get_eval_transforms

def extract_dataset_embeddings(df, model, root_dir, transform=None, device='cpu', batch_size=32):
    """
    Extracts L2-normalized embeddings for all images in df using deterministic eval transform.
    """
    if transform is None:
        transform = get_eval_transforms()
        
    model.eval()
    model.to(device)
    embeddings = []
    
    paths = [Path(root_dir) / r['image_path'] for _, r in df.iterrows()]
    
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch_tensors = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            tensor = transform(img)
            batch_tensors.append(tensor)
        batch_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            batch_embeds = model(batch_tensor).cpu().numpy()
        embeddings.append(batch_embeds)
        
    return np.vstack(embeddings)

def evaluate_retrieval(query_df, gallery_df, query_embeds, gallery_embeds):
    """
    Evaluates Task 1: Identification / Retrieval
    Returns dictionary with Recall@1, Recall@5, Recall@10, mAP, and query results.
    """
    # Cosine similarity matrix (since embeddings are L2 normalized, dot product = cosine similarity)
    sim_matrix = np.dot(query_embeds, gallery_embeds.T) # (N_query, N_gallery)
    
    recalls_at_1 = []
    recalls_at_5 = []
    recalls_at_10 = []
    aps = []
    
    query_results = []
    
    for i, (_, q_row) in enumerate(query_df.iterrows()):
        q_did = q_row['design_id']
        sims = sim_matrix[i]
        
        # Rank gallery items descending
        ranked_indices = np.argsort(-sims)
        ranked_dids = gallery_df.iloc[ranked_indices]['design_id'].values
        ranked_sims = sims[ranked_indices]
        
        matches = (ranked_dids == q_did)
        num_positives = np.sum(matches)
        
        if num_positives == 0:
            continue
            
        r1 = 1.0 if np.any(matches[:1]) else 0.0
        r5 = 1.0 if np.any(matches[:5]) else 0.0
        r10 = 1.0 if np.any(matches[:10]) else 0.0
        
        recalls_at_1.append(r1)
        recalls_at_5.append(r5)
        recalls_at_10.append(r10)
        
        # Compute Average Precision (AP)
        cum_matches = np.cumsum(matches)
        ranks = np.arange(1, len(matches) + 1)
        precisions = cum_matches / ranks
        ap = np.sum(precisions * matches) / num_positives
        aps.append(ap)
        
        query_results.append({
            'query_idx': i,
            'query_image_path': q_row['image_path'],
            'query_design_id': q_did,
            'ranked_indices': ranked_indices,
            'ranked_sims': ranked_sims,
            'matches': matches,
            'ap': ap
        })
        
    metrics = {
        'recall_at_1': float(np.mean(recalls_at_1)) if recalls_at_1 else 0.0,
        'recall_at_5': float(np.mean(recalls_at_5)) if recalls_at_5 else 0.0,
        'recall_at_10': float(np.mean(recalls_at_10)) if recalls_at_10 else 0.0,
        'mAP': float(np.mean(aps)) if aps else 0.0,
        'num_eval_queries': len(aps)
    }
    
    return metrics, sim_matrix, query_results
