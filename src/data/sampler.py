import random
import numpy as np
import torch
from torch.utils.data import Sampler
from collections import defaultdict

class DesignBatchSampler(Sampler):
    """
    Design-Aware Batch Sampler.
    Constructs batches that contain multiple images per design for multi-image groups,
    ensuring each batch has valid positive pairs for Supervised Contrastive Loss,
    without artificially duplicating physical image instances.
    """
    def __init__(self, dataset, batch_size=16, num_instances_per_design=2, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances_per_design
        self.seed = seed
        
        # Group dataset indices by label / design_id
        self.design_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            label = dataset.labels[idx]
            self.design_to_indices[label].append(idx)
            
        self.multi_designs = [lbl for lbl, idxs in self.design_to_indices.items() if len(idxs) >= 2]
        self.single_designs = [lbl for lbl, idxs in self.design_to_indices.items() if len(idxs) == 1]
        
        self.num_multi_designs = len(self.multi_designs)
        self.num_single_designs = len(self.single_designs)
        self.num_total_designs = len(self.design_to_indices)
        
    def __iter__(self):
        rng = random.Random(self.seed)
        
        # Copy indices lists
        multi_pools = {lbl: list(self.design_to_indices[lbl]) for lbl in self.multi_designs}
        for lbl in multi_pools:
            rng.shuffle(multi_pools[lbl])
            
        single_pool = list(self.single_designs)
        rng.shuffle(single_pool)
        
        batches = []
        
        # Continue constructing batches while multi-image pairs remain
        active_multi = list(self.multi_designs)
        
        while len(active_multi) > 0:
            batch = []
            rng.shuffle(active_multi)
            
            # Select 2 to 4 multi-image designs for this batch
            selected_designs = active_multi[:min(4, len(active_multi))]
            
            for did in selected_designs:
                pool = multi_pools[did]
                # Take min(num_instances, len(pool)) real physical images
                take_cnt = min(self.num_instances, len(pool))
                for _ in range(take_cnt):
                    if pool:
                        batch.append(pool.pop(0))
                if len(pool) < 2: # No more positive pairs can be formed from this design
                    if did in active_multi:
                        active_multi.remove(did)
                        
            # Fill remaining batch slots up to batch_size using singletons or remaining images
            remaining_slots = self.batch_size - len(batch)
            if remaining_slots > 0 and single_pool:
                for _ in range(min(remaining_slots, len(single_pool))):
                    s_lbl = single_pool.pop(0)
                    batch.append(self.design_to_indices[s_lbl][0])
                    
            if len(batch) > 1:
                batches.append(batch)
                
        rng.shuffle(batches)
        for b in batches:
            yield b

    def compute_batch_stats(self):
        """
        Calculates and logs batch sampler statistics over an epoch.
        """
        batches = list(self.__iter__())
        total_batches = len(batches)
        batches_with_positives = 0
        total_pos_pairs = 0
        
        for batch in batches:
            labels = [self.dataset.labels[idx] for idx in batch]
            lbl_counts = defaultdict(int)
            for l in labels:
                lbl_counts[l] += 1
            
            pos_pairs_in_batch = sum((cnt * (cnt - 1)) // 2 for cnt in lbl_counts.values() if cnt > 1)
            if pos_pairs_in_batch > 0:
                batches_with_positives += 1
            total_pos_pairs += pos_pairs_in_batch
            
        avg_pos_pairs = total_pos_pairs / total_batches if total_batches > 0 else 0.0
        pct_batches_with_pos = (batches_with_positives / total_batches * 100.0) if total_batches > 0 else 0.0
        
        stats = {
            'total_batches': total_batches,
            'total_training_designs': self.num_total_designs,
            'multi_image_designs': self.num_multi_designs,
            'singleton_designs': self.num_single_designs,
            'avg_positive_pairs_per_batch': round(avg_pos_pairs, 2),
            'pct_batches_with_at_least_one_positive': round(pct_batches_with_pos, 2)
        }
        return stats
