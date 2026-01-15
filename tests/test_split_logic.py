import pytest
import numpy as np
import pandas as pd
import os
import torch
from types import SimpleNamespace
from gen_model.dataset import MDGenDataset

@pytest.fixture
def mock_data(tmp_path):
    # Create a dummy .npy file
    protein_name = "test_prot"
    npy_dir = tmp_path / protein_name
    npy_dir.mkdir()
    npy_path = npy_dir / f"{protein_name}_R1.npy"
    
    # 100 frames, 10 residues, atom14 format (14 atoms, 3 coords)
    data = np.random.rand(100, 10, 14, 3).astype(np.float32)
    np.save(npy_path, data)
    
    # Create a split CSV
    split_csv = tmp_path / "splits.csv"
    # train_early: [0, 50)
    # train: [50, 80)
    # val: [80, 90)
    # test: [90, 100)
    df = pd.DataFrame({
        'name': [f"{protein_name}_R1"],
        'train_early_end': [50],
        'train_end': [80],
        'val_end': [90],
        'total_frames': [100]
    })
    df.to_csv(split_csv, index=False)
    
    # Create atlas CSV
    atlas_csv = tmp_path / "atlas.csv"
    atlas_df = pd.DataFrame({
        'name': [protein_name],
        'seqres': ['A' * 10]
    })
    atlas_df.to_csv(atlas_csv, index=False)
    
    args = SimpleNamespace(
        data_dir=str(tmp_path),
        suffix='',
        frame_interval=None,
        atlas_csv=str(atlas_csv),
        overfit_peptide=None,
        crop_ratio=1.0
    )
    
    return args, str(split_csv), 100

def test_train_early_mode(mock_data):
    args, split_csv, total_frames = mock_data
    # train_early should return [0, train_end) = [0, 80)
    ds = MDGenDataset(args, split_csv, mode='train_early')
    assert len(ds) == 80
    assert ds[0]['frame_indices'][0] == 0
    assert ds[79]['frame_indices'][0] == 79

def test_train_mode(mock_data):
    args, split_csv, total_frames = mock_data
    # train should return [train_early_end, train_end) = [50, 80)
    ds = MDGenDataset(args, split_csv, mode='train')
    assert len(ds) == 30
    assert ds[0]['frame_indices'][0] == 50
    assert ds[29]['frame_indices'][0] == 79

def test_val_mode(mock_data):
    args, split_csv, total_frames = mock_data
    # val should return [train_end, val_end) = [80, 90)
    ds = MDGenDataset(args, split_csv, mode='val')
    assert len(ds) == 10
    assert ds[0]['frame_indices'][0] == 80
    assert ds[9]['frame_indices'][0] == 89

def test_test_mode(mock_data):
    args, split_csv, total_frames = mock_data
    # test should return [val_end, total_frames) = [90, 100)
    ds = MDGenDataset(args, split_csv, mode='test')
    assert len(ds) == 10
    assert ds[0]['frame_indices'][0] == 90
    assert ds[9]['frame_indices'][0] == 99

def test_all_mode(mock_data):
    args, split_csv, total_frames = mock_data
    ds = MDGenDataset(args, split_csv, mode='all')
    assert len(ds) == total_frames

def test_missing_split_columns(mock_data, tmp_path):
    args, split_csv, total_frames = mock_data
    # Create CSV without the new split columns
    bad_csv = tmp_path / "bad_splits.csv"
    pd.DataFrame({'name': ['test_prot_R1'], 'total_frames': [100]}).to_csv(bad_csv, index=False)
    
    ds = MDGenDataset(args, str(bad_csv), mode='train')
    # Should fallback to all frames (100)
    assert len(ds) == total_frames
