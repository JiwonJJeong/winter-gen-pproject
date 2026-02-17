import torch
import numpy as np
import os
import sys

# Add root to path
sys.path.append('.')

from gen_model.dataset import MDGenDataset
from omegaconf import OmegaConf

def test_dataset_scaling():
    # Mock args
    args = OmegaConf.create({
        'data_dir': 'data',
        'atlas_csv': 'data/atlas.csv',
        'train_split': 'gen_model/splits/frame_splits.csv',
        'suffix': '_latent',
        'frame_interval': None,
        'crop_ratio': 0.95,
        'min_t': 0.01,
    })

    if not os.path.exists('data'):
        print("Data directory not found, skipping real test.")
        return

    try:
        dataset = MDGenDataset(args, mode='train')
        sample = dataset[0]
        
        print("\n--- Dataset Scaling Test ---")
        print(f"Calculated Coord Scale: {dataset.coord_scale:.6f}")
        
        ca_pos = sample['atom14_pos'][:, 1, :]
        mean_ca = ca_pos.mean(dim=0)
        std_ca = ca_pos.std()
        
        print(f"Mean CA Position after centering: {mean_ca.tolist()} (Scale: {dataset.coord_scale:.4f})")
        print(f"Std CA Position after scaling: {std_ca.item():.6f}")
        
        if torch.abs(mean_ca).max() < 1e-5:
            print("SUCCESS: Centering works.")
        else:
            print("FAILURE: Centering failed.")
            
        # Standard deviation should be roughly 1.0 (it's exactly 1.0 for the sampled subset used for calculation)
        print(f"Max coordinate: {sample['atom14_pos'].abs().max().item():.3f}")
        
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test_dataset_scaling()
