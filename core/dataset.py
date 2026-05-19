import torch
import numpy as np
import glob
import os
from torch.utils.data import Dataset, DataLoader

class CongestionDataset(Dataset):
    """
    Custom Dataset for loading pre-processed VLSI layout shards (.npz).
    Each shard typically contains 'X' (features), 'Y' (congestion), and 'mask'.
    """
    def __init__(self, shard_paths, is_binary=False, threshold=0.1):
        self.shard_paths = sorted(shard_paths)
        self.is_binary = is_binary
        self.threshold = threshold
        
        # We load metadata to know how many samples are in each shard
        self.samples_per_shard = []
        self._calculate_total_samples()

    def _calculate_total_samples(self):
        # We assume shards are uniform in size (e.g., 100 samples per shard)
        # to avoid opening every file at initialization.
        if self.shard_paths:
            with np.load(self.shard_paths[0]) as data:
                self.shard_size = data['X'].shape[0]
            self.total_samples = len(self.shard_paths) * self.shard_size

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        shard_idx = idx // self.shard_size
        sample_idx = idx % self.shard_size
        
        with np.load(self.shard_paths[shard_idx]) as data:
            # Features: [Macro, Cell Density, RUDY, RUDY_std]
            # Selecting indices 0, 1, 2, and 5 based on your data structure
            x = data['X'][sample_idx, [0, 1, 2, 5]]
            
            # Target: Horizontal (2) and Vertical (3) routing
            y = data['Y'][sample_idx, [2, 3]]
            
            mask = data['mask'][sample_idx]

        # Convert to Binary if experiment requires (Exp 5 & 6)
        if self.is_binary:
            y = (y > self.threshold).astype(np.float32)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(mask).float()
        )

def get_dataloader(data_dir, batch_size=16, split='train', is_binary=False):
    """
    Utility to create a DataLoader from a directory of .npz shards.
    """
    all_shards = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    
    # Standard Split Strategy used in the project
    if split == 'train':
        shards = all_shards[:80]
    elif split == 'val':
        shards = all_shards[80:92]
    else: # test
        shards = all_shards[92:]
        
    dataset = CongestionDataset(shards, is_binary=is_binary)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
