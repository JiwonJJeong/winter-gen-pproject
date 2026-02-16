"""Test script to verify MDGenDataset outputs correct features for SE(3) diffusion."""
import torch
import numpy as np
from omegaconf import OmegaConf
import sys
sys.path.insert(0, '.')

from gen_model.dataset import MDGenDataset
from data_se3.data.se3_diffuser import SE3Diffuser


def test_dataset():
    """Test that MDGenDataset outputs all required features."""
    
    # 1. Create SE3Diffuser
    print("Creating SE3Diffuser...")
    se3_conf = OmegaConf.create({
        'diffuse_rot': True,
        'diffuse_trans': True,
        'so3': {
            'schedule': 'logarithmic',
            'min_sigma': 0.1,
            'max_sigma': 1.5,
            'num_sigma': 1000,
            'cache_dir': './',
        },
        'r3': {
            'min_b': 0.1,
            'max_b': 20.0,
            'coordinate_scaling': 0.1,  # Scale Angstroms to ~1.0 range
        }
    })
    diffuser = SE3Diffuser(se3_conf)
    print("✓ SE3Diffuser created")
    
    # 2. Create dataset args
    print("\nCreating dataset...")
    args = OmegaConf.create({
        'data_dir': 'data',  # Update with your actual data directory
        'atlas_csv': 'data/atlas.csv',  # Update with your actual atlas path
        'suffix': '_latent',
        'frame_interval': None,
        'crop_ratio': 0.95,
        'min_t': 0.01,
        'overfit_peptide': None,
    })
    
    # 3. Create dataset
    dataset = MDGenDataset(
        args=args,
        diffuser=diffuser,
        split='gen_model/splits/frame_splits.csv',  # Update with your actual split
        mode='train',
        repeat=1,
        num_consecutive=1,
        stride=1
    )
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # 4. Get a sample
    print("\nTesting dataset output...")
    try:
        sample = dataset[0]
        print("✓ Successfully loaded sample")
    except Exception as e:
        print(f"✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Check required fields
    required_fields = [
        'aatype', 'seq_idx', 'chain_idx', 'res_mask', 'fixed_mask',
        'rigids_0', 'atom37_pos', 'atom14_pos', 'torsion_angles_sin_cos',
        'rigids_t', 'rot_score', 'trans_score', 'rot_score_scaling', 'trans_score_scaling',
        't', 'sc_ca_t', 'residx_atom14_to_atom37', 'atom37_mask', 'residue_index'
    ]
    
    print("\nChecking required fields:")
    missing_fields = []
    for field in required_fields:
        if field in sample:
            print(f"  ✓ {field}: {sample[field].shape if hasattr(sample[field], 'shape') else type(sample[field])}")
        else:
            print(f"  ✗ {field}: MISSING")
            missing_fields.append(field)
    
    if missing_fields:
        print(f"\n✗ Missing fields: {missing_fields}")
        return
    
    # 6. Check shapes
    print("\nValidating tensor shapes:")
    L = sample['aatype'].shape[0]
    print(f"  Sequence length L = {L}")
    
    shape_checks = {
        'aatype': (L,),
        'seq_idx': (L,),
        'chain_idx': (L,),
        'res_mask': (L,),
        'fixed_mask': (L,),
        'rigids_0': (L, 7),
        'atom37_pos': (L, 37, 3),
        'atom14_pos': (L, 14, 3),
        'torsion_angles_sin_cos': (L, 7, 2),
        'rigids_t': (L, 7),
        'rot_score': (L, 3),
        'trans_score': (L, 3),
        'sc_ca_t': (L, 3),
        'residx_atom14_to_atom37': (L, 14),
        'atom37_mask': (L, 37),
        'residue_index': (L,),
    }
    
    all_correct = True
    for field, expected_shape in shape_checks.items():
        actual_shape = tuple(sample[field].shape)
        if actual_shape == expected_shape:
            print(f"  ✓ {field}: {actual_shape}")
        else:
            print(f"  ✗ {field}: expected {expected_shape}, got {actual_shape}")
            all_correct = False
    
    # 7. Check scalar fields
    print("\nChecking scalar fields:")
    scalar_fields = ['t', 'rot_score_scaling', 'trans_score_scaling']
    for field in scalar_fields:
        value = sample[field]
        if isinstance(value, (float, int)) or (torch.is_tensor(value) and value.numel() == 1):
            print(f"  ✓ {field}: {value}")
        else:
            print(f"  ✗ {field}: expected scalar, got {type(value)}")
            all_correct = False
    
    # 8. Summary
    print("\n" + "="*60)
    if all_correct and not missing_fields:
        print("✓ ALL CHECKS PASSED! Dataset is ready for SE(3) diffusion training.")
    else:
        print("✗ Some checks failed. Please review the errors above.")
    print("="*60)


if __name__ == '__main__':
    test_dataset()
