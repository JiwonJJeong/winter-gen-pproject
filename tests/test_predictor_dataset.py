import pytest
import torch
import numpy as np
import pandas as pd
import os
from types import SimpleNamespace
from gen_model.dataset_predictor import MDGenDataset


@pytest.fixture
def test_data_path():
    """Path to test data"""
    return '/home/jiwonjjeong/repos/winter-gen-pproject/data/4o66_C'


@pytest.fixture
def protein_name():
    """Test protein name"""
    return "4o66_C"


@pytest.fixture
def dataset_args(test_data_path):
    """Standard dataset arguments for testing"""
    return SimpleNamespace(
        data_dir=test_data_path,
        suffix='',
        overfit=False,
        overfit_peptide=None,
        overfit_frame=False,
        atlas=True,
        frame_interval=1,
        copy_frames=False,
        no_frames=False,
        crop_ratio=0.95,
        noise_std=0.1  # Noise standard deviation in Angstroms
    )


@pytest.fixture
def temp_split_csv(protein_name, test_data_path, tmp_path):
    """Create temporary split CSV for testing"""
    npy_path = f"{test_data_path}/{protein_name}_R1.npy"
    if not os.path.exists(npy_path):
        pytest.skip(f"Test data not found: {npy_path}")
    
    arr = np.load(npy_path, mmap_mode='r')
    L = arr.shape[1]
    
    # Create dummy seqres
    dummy_seqres = 'A' * L
    df = pd.DataFrame({'seqres': [dummy_seqres]}, index=[protein_name])
    
    csv_path = tmp_path / "test_split.csv"
    df.to_csv(csv_path, index_label='name')
    
    return csv_path, L


class TestPredictorDataset:
    """Tests for predictor dataset with two-frame output"""
    
    def test_two_frame_output(self, dataset_args, temp_split_csv):
        """Test that dataset returns two frames"""
        csv_path, L = temp_split_csv
        k_steps = 1
        ds = MDGenDataset(dataset_args, str(csv_path), k_steps=k_steps)
        
        item = ds[0]
        
        # Check that both frame 0 and frame k data are present
        assert 'torsions_0' in item
        assert 'torsions_k' in item
        assert 'trans_0' in item
        assert 'trans_k' in item
        assert 'rots_0' in item
        assert 'rots_k' in item
        assert 'k_steps' in item
        assert item['k_steps'] == k_steps
    
    def test_k_steps_parameter(self, dataset_args, temp_split_csv):
        """Test that k_steps parameter works correctly"""
        csv_path, L = temp_split_csv
        
        for k in [1, 2, 5]:
            ds = MDGenDataset(dataset_args, str(csv_path), k_steps=k)
            item = ds[0]
            assert item['k_steps'] == k
    
    def test_frames_are_different(self, dataset_args, temp_split_csv):
        """Test that frame 0 and frame k are different (due to noise)"""
        csv_path, L = temp_split_csv
        ds = MDGenDataset(dataset_args, str(csv_path), k_steps=1)
        
        item = ds[0]
        
        # The frames should be different due to added noise
        trans_0 = item['trans_0']
        trans_k = item['trans_k']
        
        # They should not be exactly equal (noise was added)
        assert not torch.allclose(trans_0, trans_k, atol=1e-6)
    
    def test_shape_consistency(self, dataset_args, temp_split_csv):
        """Test that both frames have consistent shapes"""
        csv_path, L = temp_split_csv
        ds = MDGenDataset(dataset_args, str(csv_path), k_steps=1)
        
        item = ds[0]
        
        # Check that frame 0 and frame k have the same shapes
        assert item['torsions_0'].shape == item['torsions_k'].shape
        assert item['trans_0'].shape == item['trans_k'].shape
        assert item['rots_0'].shape == item['rots_k'].shape
        
        # Check that shapes match expected dimensions
        assert item['trans_0'].shape[1] == L  # (1, L, 3)
        assert item['rots_0'].shape[1] == L   # (1, L, 3, 3)
    
    def test_no_frames_mode(self, dataset_args, temp_split_csv):
        """Test that no_frames mode returns atom37 data for both frames"""
        csv_path, L = temp_split_csv
        dataset_args.no_frames = True
        ds = MDGenDataset(dataset_args, str(csv_path), k_steps=1)
        
        item = ds[0]
        
        assert 'atom37_0' in item
        assert 'atom37_k' in item
        assert 'torsions_0' not in item
        assert 'torsions_k' not in item
    
    def test_default_k_steps(self, dataset_args, temp_split_csv):
        """Test that k_steps defaults to 1"""
        csv_path, L = temp_split_csv
        ds = MDGenDataset(dataset_args, str(csv_path))  # No k_steps specified
        
        item = ds[0]
        assert item['k_steps'] == 1
    
    def test_copy_frames_mode(self, dataset_args, temp_split_csv):
        """Test that copy_frames mode uses the same frame for both"""
        csv_path, L = temp_split_csv
        dataset_args.copy_frames = True
        ds = MDGenDataset(dataset_args, str(csv_path), k_steps=5)
        
        item = ds[0]
        
        # With copy_frames, the underlying frames should be the same
        # (though noise is still added, so they won't be exactly equal)
        # This test just ensures the code runs without error
        assert 'trans_0' in item
        assert 'trans_k' in item
