import numpy as np
import pandas as pd

def compute_roc_auc(y_true, y_scores):
    """
    Computes ROC-AUC score using pure NumPy to avoid external dependencies.
    """
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
        
    # Rank scores
    order = np.argsort(y_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1)
    
    # Handle ties in scores
    sorted_scores = y_scores[order]
    unique_scores, idx, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    for i in range(len(unique_scores)):
        if counts[i] > 1:
            tie_mask = (idx == i)
            ranks[order[tie_mask]] = np.mean(ranks[order[tie_mask]])
            
    pos_rank_sum = np.sum(ranks[pos_mask])
    auc = (pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)
    return float(auc)

def evaluate_verification(query_df, gallery_df, sim_matrix, threshold=0.5):
    """
    Evaluates Task 2: Pairwise Verification
    Pairs query i with gallery j.
    """
    y_true = []
    y_scores = []
    
    for i, (_, q_row) in enumerate(query_df.iterrows()):
        q_did = q_row['design_id']
        for j, (_, g_row) in enumerate(gallery_df.iterrows()):
            g_did = g_row['design_id']
            is_same = 1 if (q_did == g_did) else 0
            score = sim_matrix[i, j]
            y_true.append(is_same)
            y_scores.append(score)
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    roc_auc = compute_roc_auc(y_true, y_scores)
    
    y_pred = (y_scores >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return {
        'roc_auc': float(roc_auc),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'operating_threshold': float(threshold),
        'pos_pairs_count': int(np.sum(y_true == 1)),
        'neg_pairs_count': int(np.sum(y_true == 0))
    }

def find_optimal_threshold(val_query_df, val_gallery_df, val_sim_matrix):
    """
    Finds threshold on validation set that maximizes F1 score.
    """
    best_f1 = -1
    best_thresh = 0.5
    for thresh in np.linspace(0.0, 1.0, 101):
        res = evaluate_verification(val_query_df, val_gallery_df, val_sim_matrix, threshold=thresh)
        if res['f1'] > best_f1:
            best_f1 = res['f1']
            best_thresh = thresh
    return float(best_thresh)
