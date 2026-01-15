import torch
import numpy as np
from types import SimpleNamespace
from gen_model.dataset import MDGenDataset
import pandas as pd
import os

def test_balanced_masking_uniformity():
    # Setup mock data: Frame 0-4 are linear, Frame 5+ are bent (L-shape)
    L = 50
    num_frames = 30
    coords = np.zeros((num_frames, L, 14, 3))
    
    # Linear conformation (Frame 0)
    coords[:5, :, 1, 0] = np.arange(L) 
    
    # L-shape conformation (Frame 5+)
    # First 25 residues on X axis, next 25 on Y axis
    coords[5:, :25, 1, 0] = np.arange(25)
    coords[5:, 25:, 1, 0] = 24 # fixed x
    coords[5:, 25:, 1, 1] = np.arange(1, 26) # increasing y
    
    # Save dummy npy
    data_dir = "tmp_data"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/prot_bal", exist_ok=True)
    np.save(f"{data_dir}/prot_bal/prot_bal_R1.npy", coords)
    
    # Mock args
    args = SimpleNamespace(
        data_dir=data_dir,
        atlas_csv="tmp_atlas.csv",
        crop_ratio=0.2, 
        pep_name=None,
        replica=None,
        suffix='',
        frame_interval=None,
        overfit_peptide=None
    )
    
    # Mock atlas and splits
    pd.DataFrame({'name': ['prot_bal'], 'seqres': ['A'*L]}).to_csv("tmp_atlas.csv", index=False)
    split_df = pd.DataFrame({
        'name': ['prot_bal_R1'],
        'train_early_end': [5],  # Train starts at frame 5 (L-shape)
        'train_end': [15],
        'val_end': [25]
    })
    split_df.to_csv("tmp_splits.csv", index=False)
    
    # Instantiate dataset
    # Test 1: Train Early (Should use Frame 0 - Linear)
    ds_early = MDGenDataset(args, split="tmp_splits.csv", mode='train_early')
    weights_early = ds_early.balanced_weights['prot_bal']
    
    # Test 2: Train (Should use Frame 5 - L-shape)
    ds_train = MDGenDataset(args, split="tmp_splits.csv", mode='train')
    weights_train = ds_train.balanced_weights['prot_bal']
    
    # Compare weights - they should be significantly different
    # Linear: Endpoints have fewer neighbors. 
    # L-shape: Corner residues might have different neighbor counts.
    # Just asserting they are not equal proves we used different frames.
    assert not np.allclose(weights_early, weights_train), "Train and Train Early should use different reference frames!"
    
    # Verify Val mode has no weights
    ds_val = MDGenDataset(args, split="tmp_splits.csv", mode='val')
    assert 'prot_bal' not in ds_val.balanced_weights

    print("Verification Successful: Modes use correct reference frames.")
    
    # Cleanup
    os.remove("tmp_atlas.csv")
    os.remove("tmp_splits.csv")
    import shutil
    shutil.rmtree(data_dir)

if __name__ == "__main__":
    test_balanced_masking_uniformity()
