import os
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.color_invariant_net import ColorInvariantResNet50
from src.data.augmentations import get_train_transforms, get_eval_transforms
from src.data.dataset import SareeDataset
from src.data.sampler import DesignBatchSampler
from src.losses.supcon import SupConLoss
from src.evaluation.retrieval import evaluate_retrieval, extract_dataset_embeddings

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def train_color_invariant_model(
    root_dir,
    output_dir,
    embed_dim=256,
    freeze_backbone=False,
    unfreeze_from_layer='layer4',
    use_color_aug=True,
    batch_size=16,
    epochs=35,
    lr_backbone=1.5e-5,
    lr_head=3e-4,
    temperature=0.07,
    seed=42,
    device='cpu'
):
    """
    Trains the proposed Color-Invariant Metric Learning Model using SupConLoss and DesignBatchSampler.
    Supports toggling color augmentations via use_color_aug.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dir = root_dir / "data" / "splits"
    train_df = pd.read_csv(splits_dir / "primary_train.csv")
    val_df = pd.read_csv(splits_dir / "primary_val.csv")
    
    # Create Datasets
    train_transforms = get_train_transforms(use_color_aug=use_color_aug)
    eval_transforms = get_eval_transforms()
    
    train_dataset = SareeDataset(train_df, root_dir, transform=train_transforms)
    
    # Design Batch Sampler for training
    batch_sampler = DesignBatchSampler(
        train_dataset,
        batch_size=batch_size,
        num_instances_per_design=2,
        seed=seed
    )
    
    sampler_stats = batch_sampler.compute_batch_stats()
    
    # Validation split format for evaluation
    val_query_rows = []
    val_gallery_rows = []
    for did, grp in val_df.groupby('design_id'):
        sorted_grp = grp.sort_values('colorway_id')
        val_query_rows.append(sorted_grp.iloc[0])
        for _, r in sorted_grp.iloc[1:].iterrows():
            val_gallery_rows.append(r)
            
    val_query_df = pd.DataFrame(val_query_rows)
    val_gallery_df = pd.DataFrame(val_gallery_rows)
    
    val_query_dataset = SareeDataset(val_query_df, root_dir, transform=eval_transforms)
    val_gallery_dataset = SareeDataset(val_gallery_df, root_dir, transform=eval_transforms)
    
    # Initialize Model
    model = ColorInvariantResNet50(
        embed_dim=embed_dim,
        freeze_backbone=freeze_backbone,
        unfreeze_from_layer=unfreeze_from_layer
    ).to(device)
    
    total_params, trainable_params = count_parameters(model)
    
    # Differential learning rates for backbone vs projection head
    if freeze_backbone:
        params = [{'params': model.projection_head.parameters(), 'lr': lr_head}]
    else:
        params = [
            {'params': model.backbone.parameters(), 'lr': lr_backbone},
            {'params': model.projection_head.parameters(), 'lr': lr_head}
        ]
        
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    criterion = SupConLoss(temperature=temperature)
    
    best_val_map = -1.0
    best_epoch = 0
    checkpoint_path = output_dir / "best_model.pth"
    
    history = []
    
    print(f"\nStarting training for {epochs} epochs on {device} (Total Params: {total_params:,}, Trainable: {trainable_params:,})...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0):
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            embeds = model(imgs)
            loss = criterion(embeds, labels)
            
            if loss.requires_grad and not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Validation Evaluation
        model.eval()
        val_q_embeds = extract_dataset_embeddings(val_query_df, model, root_dir, device=device)
        val_g_embeds = extract_dataset_embeddings(val_gallery_df, model, root_dir, device=device)
        val_metrics, _, _ = evaluate_retrieval(val_query_df, val_gallery_df, val_q_embeds, val_g_embeds)
        
        val_map = val_metrics['mAP']
        val_r1 = val_metrics['recall_at_1']
        
        history.append({
            'epoch': epoch,
            'train_loss': round(avg_loss, 4),
            'val_mAP': round(val_map, 4),
            'val_R1': round(val_r1, 4)
        })
        
        if val_map > best_val_map:
            best_val_map = val_map
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'embed_dim': embed_dim,
                'val_mAP': val_map,
                'val_R1': val_r1,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'sampler_stats': sampler_stats
            }, checkpoint_path)
            
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s. Best Epoch: {best_epoch} with Val mAP = {best_val_map:.4f}")
    
    meta_info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_time_seconds': round(elapsed, 2),
        'best_epoch': best_epoch,
        'best_val_mAP': round(best_val_map, 4),
        'total_epochs': epochs
    }
    
    return checkpoint_path, sampler_stats, meta_info
