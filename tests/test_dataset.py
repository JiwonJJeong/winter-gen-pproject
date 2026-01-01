import os
import sys
import torch
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add current dir to path to import gen_model
sys.path.append(os.getcwd())

from gen_model.dataset import MDGenDataset

class Args:
    """Mock arguments for testing."""
    def __init__(self, pep_name=None, pep_seq=None, train_frame_limit=None):
        self.pep_name = pep_name
        self.pep_seq = pep_seq
        self.train_frame_limit = train_frame_limit
        self.data_dir = "data"
        self.suffix = ""
        self.num_frames = 50
        self.frame_interval = None
        self.overfit = False
        self.overfit_peptide = None
        self.overfit_frame = False
        self.atlas = False
        self.copy_frames = False
        self.no_frames = False
        self.no_pad = False
        self.short_md = False
        self.crop = 256

class RigidMock:
    """Mock for the Rigid object returned by atom14_to_frames."""
    def __init__(self, shape):
        self._trans = torch.zeros((*shape, 3))
        self._rots = MagicMock()
        self._rots._rot_mats = torch.zeros((*shape, 3, 3))
    
    @property
    def shape(self):
        return (50, 5) # dummy length

def run_tests():
    # Setup common mocks
    with patch('gen_model.dataset.atom14_to_frames', return_value=RigidMock((50, 5))):
        with patch('gen_model.dataset.atom14_to_atom37', return_value=np.zeros((50, 5, 37, 3))):
            with patch('gen_model.dataset.atom37_to_torsions', return_value=(torch.zeros((50, 5, 7, 2)), torch.zeros((5, 7)))):
                with patch('numpy.lib.format.open_memmap', return_value=np.zeros((200, 5, 14, 3))):
                    
                    split_file = "gen_model/splits/atlas.csv"
                    
                    # --- Test 1: Protein ID lookup in CSV ---
                    print("\nTest 1: Protein ID lookup (1a62_A)")
                    args1 = Args(pep_name="1a62_A")
                    dataset1 = MDGenDataset(args1, split=split_file, is_train=True)
                    item1 = dataset1[0]
                    # Sequence for 1a62_A starts with MNLTE... (indices 12, 2, 10, 16, 6)
                    print(f"Sequence indices (first 5): {item1['seqres'][:5]}")
                    assert item1['seqres'][0] == 12, "Should have looked up M (12) from CSV"
                    
                    # --- Test 2: Fallback to sequence ---
                    print("\nTest 2: Fallback to sequence (AAAAA)")
                    args2 = Args(pep_name="AAAAA")
                    dataset2 = MDGenDataset(args2, split=split_file, is_train=True)
                    item2 = dataset2[0]
                    print(f"Sequence indices: {item2['seqres']}")
                    assert all(item2['seqres'] == 0), "Should have used Alanines (0)"
                    
                    # --- Test 3: Explicit pep_seq override ---
                    print("\nTest 3: Explicit pep_seq override (RRRRR)")
                    args3 = Args(pep_name="1a62_A", pep_seq="RRRRR")
                    dataset3 = MDGenDataset(args3, split=split_file, is_train=True)
                    item3 = dataset3[0]
                    print(f"Sequence indices: {item3['seqres']}")
                    assert all(item3['seqres'] == 1), "Should have used Arginines (1)"

                    # --- Test 4: Temporal Window Logic ---
                    print("\nTest 4: Temporal window (limit=100)")
                    args4 = Args(pep_name="AAAAA", train_frame_limit=100)
                    dataset4 = MDGenDataset(args4, split=split_file, is_train=True)
                    frame_starts = [dataset4[0]['frame_start'] for _ in range(100)]
                    max_start = max(frame_starts)
                    print(f"Max frame_start with limit: {max_start}")
                    assert max_start <= 100 - 50, f"Frame start {max_start} exceeded limit"

                    print("\nTest 5: Validation has no limit")
                    dataset5 = MDGenDataset(args4, split=split_file, is_train=False)
                    frame_starts_val = [dataset5[0]['frame_start'] for _ in range(100)]
                    max_start_val = max(frame_starts_val)
                    print(f"Max frame_start in validation: {max_start_val}")
                    assert max_start_val > 50, "Validation should sample beyond the limit"

    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    run_tests()
