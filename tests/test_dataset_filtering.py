import pytest
import torch
import numpy as np
import pandas as pd
import os
from types import SimpleNamespace
from gen_model.dataset import MDGenDataset

@pytest.fixture
def mock_setup(tmp_path):
    # 1. Create dummy data directories
    data_dir = tmp_path / "data"
    p1_dir = data_dir / "prot1"
    p2_dir = data_dir / "prot2"
    p1_dir.mkdir(parents=True)
    p2_dir.mkdir(parents=True)
    
    # 2. Create dummy npy files
    # prot1 has 10 frames, length 5
    p1_data = np.zeros((10, 5, 14, 3))
    np.save(p1_dir / "prot1_R1.npy", p1_data)
    np.save(p1_dir / "prot1_R2.npy", p1_data) # Add R2 for testing
    
    # prot2 has 5 frames, length 8
    p2_data = np.zeros((5, 8, 14, 3))
    np.save(p2_dir / "prot2_R1.npy", p2_data)
    
    # 3. Create dummy split CSV
    split_df = pd.DataFrame({
        'name': ['prot1_R1', 'prot1_R2', 'prot2_R1'],
        'train_early_end': [2, 2, 1],
        'train_end': [8, 8, 4],
        'val_end': [10, 10, 5]
    })
    split_csv = tmp_path / "splits.csv"
    split_df.to_csv(split_csv, index=False)
    
    # 4. Create dummy atlas CSV (for sequence mapping)
    atlas_df = pd.DataFrame({
        'name': ['prot1', 'prot2'],
        'seqres': ['AAAAA', 'VVVVVVVV']
    })
    atlas_csv = tmp_path / "atlas.csv"
    atlas_df.to_csv(atlas_csv, index=False)
    
    args = SimpleNamespace(
        data_dir=str(data_dir),
        atlas_csv=str(atlas_csv),
        suffix='',
        frame_interval=None,
        overfit_peptide=None,
        pep_name=None,
        replica=1 # Default
    )
    
    return args, str(split_csv)

def test_filtering_by_pep_name(mock_setup):
    args, split_csv = mock_setup
    
    # Test with prot1 only
    args.pep_name = "prot1"
    ds = MDGenDataset(args, split=split_csv, mode='all')
    
    # Should only have prot1 frames
    assert len(ds.frame_index) == 10
    for protein_idx, frame_idx in ds.frame_index:
        name = ds.df.index[protein_idx]
        assert name.startswith("prot1")
        
    # Check item length
    item = ds[0]
    assert item['seqres'].shape[0] == 5
    assert item['mask'].shape[0] == 5

def test_filtering_by_pep_name_prot2(mock_setup):
    args, split_csv = mock_setup
    
    # Test with prot2 only
    args.pep_name = "prot2"
    ds = MDGenDataset(args, split=split_csv, mode='all')
    
    # Should only have prot2 frames
    assert len(ds.frame_index) == 5
    for protein_idx, frame_idx in ds.frame_index:
        name = ds.df.index[protein_idx]
        assert name.startswith("prot2")

    # Check item length
    item = ds[0]
    assert item['seqres'].shape[0] == 8
    assert item['mask'].shape[0] == 8


def test_filtering_by_replica(mock_setup):
    args, split_csv = mock_setup
    
    # Test with prot1 and replica 2
    args.pep_name = "prot1"
    args.replica = 2
    ds = MDGenDataset(args, split=split_csv, mode='all')
    
    # Should only have prot1_R2 frames
    assert len(ds.frame_index) == 10
    for protein_idx, frame_idx in ds.frame_index:
        name = ds.df.index[protein_idx]
        assert name == "prot1_R2"

def test_filtering_by_replica_default(mock_setup):
    args, split_csv = mock_setup
    
    # By default, should only load _R1 even if _R2 exists
    args.pep_name = "prot1"
    args.replica = 1
    ds = MDGenDataset(args, split=split_csv, mode='all')
    
    assert len(ds.frame_index) == 10
    for protein_idx, frame_idx in ds.frame_index:
        name = ds.df.index[protein_idx]
        assert name == "prot1_R1"

def test_no_filtering_loads_all_if_replica_is_none(mock_setup):
    args, split_csv = mock_setup
    
    # Test without pep_name and replica=None
    args.pep_name = None
    args.replica = None
    ds = MDGenDataset(args, split=split_csv, mode='all')
    
    # Should have all (10 from R1, 10 from R2, 5 from p2_R1)
    assert len(ds.frame_index) == 25
