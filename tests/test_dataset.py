import pytest
import torch
import numpy as np
import pandas as pd
import os
from types import SimpleNamespace
from gen_model.dataset_projector import MDGenDataset
from gen_model.dataset_predictor import MDGenDataset as PredictorDataset
from gen_model.dataset_interpolator import MDGenDataset as InterpolatorDataset


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
        num_frames=5,
        overfit=False,
        overfit_peptide=None,
        overfit_frame=False,
        atlas=True,
        frame_interval=1,
        copy_frames=False,
        no_frames=False,
        crop_ratio=0.95  # Keep 95%, remove 5%
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


class TestSpatialMasking:
    """Tests for spatial masking implementation"""
    
    def test_mask_percentage(self, dataset_args, temp_split_csv):
        """Test that exactly 5% of residues are removed"""
        csv_path, L = temp_split_csv
        ds = MDGenDataset(dataset_args, str(csv_path))
        
        item = ds[0]
        mask = item['mask']
        
        expected_keep = int(L * dataset_args.crop_ratio)
        actual_keep = np.sum(mask)
        removed_count = L - actual_keep
        removed_pct = (removed_count / L) * 100
        
        assert abs(actual_keep - expected_keep) <= 1, \
            f"Expected to keep {expected_keep} residues, got {actual_keep}"
        assert 4.0 <= removed_pct <= 6.0, \
            f"Expected ~5% removal, got {removed_pct:.1f}%"
    
    def test_shape_preservation(self, dataset_args, temp_split_csv):
        """Test that all tensors maintain full length (no slicing)"""
        csv_path, L = temp_split_csv
        ds = MDGenDataset(dataset_args, str(csv_path))
        
        item = ds[0]
        
        assert item['torsions'].shape[1] == L, "torsions was sliced"
        assert item['trans'].shape[1] == L, "trans was sliced"
        assert item['rots'].shape[1] == L, "rots was sliced"
        assert item['mask'].shape[0] == L, "mask was sliced"
        assert item['torsion_mask'].shape[0] == L, "torsion_mask was sliced"
    
    def test_torsion_mask_independence(self, dataset_args, temp_split_csv):
        """Test that torsion_mask is NOT affected by spatial masking"""
        csv_path, L = temp_split_csv
        ds = MDGenDataset(dataset_args, str(csv_path))
        
        item = ds[0]
        mask = item['mask']
        torsion_mask = item['torsion_mask']
        
        # Find spatially masked residues
        masked_residues = np.where(mask == 0)[0]
        
        if len(masked_residues) > 0:
            # torsion_mask should still reflect chemical validity
            # (not all zeros for masked residues)
            sample_idx = masked_residues[0]
            has_valid_torsions = torch.any(torsion_mask[sample_idx, :] == 1)
            assert has_valid_torsions, \
                f"torsion_mask was incorrectly zeroed for masked residue {sample_idx}"
    
    def test_no_data_zeroing(self, dataset_args, temp_split_csv):
        """Test that data values are preserved (not zeroed) for masked residues"""
        csv_path, L = temp_split_csv
        ds = MDGenDataset(dataset_args, str(csv_path))
        
        item = ds[0]
        mask = item['mask']
        torsions = item['torsions']
        
        masked_residues = np.where(mask == 0)[0]
        
        if len(masked_residues) > 0:
            # Check that torsion values are not all zero
            sample_idx = masked_residues[0]
            torsion_values = torsions[:, sample_idx, :, :]
            has_nonzero = torch.any(torsion_values != 0)
            assert has_nonzero, \
                f"Data was incorrectly zeroed for masked residue {sample_idx}"
    
    def test_torsion_mask_shape(self, dataset_args, temp_split_csv):
        """Test that torsion_mask has correct shape [L, 7]"""
        csv_path, L = temp_split_csv
        ds = MDGenDataset(dataset_args, str(csv_path))
        
        item = ds[0]
        torsion_mask = item['torsion_mask']
        
        assert torsion_mask.shape == (L, 7), \
            f"Expected torsion_mask shape ({L}, 7), got {torsion_mask.shape}"
    
    def test_randomness(self, dataset_args, temp_split_csv):
        """Test that different samples produce different masks"""
        csv_path, L = temp_split_csv
        ds = MDGenDataset(dataset_args, str(csv_path))
        
        masks = []
        for _ in range(3):
            item = ds[0]
            masks.append(item['mask'].copy())
        
        # At least one pair should be different (very high probability)
        different = False
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if not np.array_equal(masks[i], masks[j]):
                    different = True
                    break
        
        assert different, "All masks are identical - randomness not working"
    
    @pytest.mark.parametrize("dataset_class,name", [
        (MDGenDataset, "projector"),
        (PredictorDataset, "predictor"),
        (InterpolatorDataset, "interpolator"),
    ])
    def test_all_dataset_variants(self, dataset_class, name, dataset_args, temp_split_csv):
        """Test that all dataset variants implement spatial masking correctly"""
        csv_path, L = temp_split_csv
        ds = dataset_class(dataset_args, str(csv_path))
        
        item = ds[0]
        mask = item['mask']
        
        expected_keep = int(L * dataset_args.crop_ratio)
        actual_keep = np.sum(mask)
        
        assert abs(actual_keep - expected_keep) <= 1, \
            f"{name} dataset: Expected {expected_keep}, got {actual_keep}"
