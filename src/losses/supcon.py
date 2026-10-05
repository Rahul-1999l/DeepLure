import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Computes contrastive loss over positive pairs of the same design_id.
    Excludes the anchor itself from positive matches.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (batch_size, embed_dim) - L2 normalized embeddings
        labels: (batch_size,) - Integer design class labels
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        labels = labels.view(-1, 1)
        # Binary mask: mask[i, j] = 1 if labels[i] == labels[j]
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Self-contrast mask: logits_mask[i, j] = 1 for all i != j
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask # Exclude anchor self-match
        
        # Compute dot product cosine similarity matrix divided by temperature
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # For numerical stability, subtract max logit per row
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask out self-contrast from denominator
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean log-likelihood for positive pairs per anchor
        pos_per_anchor = mask.sum(1)
        valid_anchors = (pos_per_anchor > 0)
        
        if not torch.any(valid_anchors):
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / pos_per_anchor[valid_anchors]
        loss = -mean_log_prob_pos.mean()
        
        return loss
