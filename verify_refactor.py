import torch
import numpy as np
from gen_model.dataset import MDGenDataset
import unittest.mock as mock

# Mock open_memmap
mock_arr = np.random.randn(200, 10, 14, 3).astype(np.float32)
patcher = mock.patch('numpy.lib.format.open_memmap', return_value=mock_arr)
patcher.start()

# Mock geometry functions that might fail on random input
mock_frames = mock.MagicMock()
mock_frames.shape = (200, 10)
mock_frames._trans = torch.randn(200, 10, 3)
mock_frames._rots._rot_mats = torch.randn(200, 10, 3, 3)
patcher2 = mock.patch('gen_model.dataset.atom14_to_frames', return_value=mock_frames)
patcher2.start()

patcher3 = mock.patch('gen_model.dataset.atom14_to_atom37', return_value=np.random.randn(200, 10, 37, 3))
patcher3.start()

patcher4 = mock.patch('gen_model.dataset.atom37_to_torsions', return_value=(torch.randn(200, 10, 7, 2), torch.ones(200, 10, 7)))
patcher4.start()

import sys
import os

# Mock args
class Args:
    def __init__(self):
        self.pep_name = "AAAAA" # Valid sequence
        self.data_dir = "data"
        self.suffix = ""
        self.num_frames = 50
        self.atlas = False
        self.frame_interval = None
        self.overfit = False
        self.overfit_peptide = None
        self.overfit_frame = False
        self.copy_frames = False
        self.no_frames = False
        self.crop = 256
        self.train_frame_limit = 100

args = Args()

# Test 1: Initialization with pep_name
print("Testing initialization with pep_name...")
dataset = MDGenDataset(args, is_train=True)
print(f"Dataset length: {len(dataset)}")
assert len(dataset) == 1000

# Test 2: Get item and check frame_start limit
print("Testing frame_start limit for training...")
frame_starts = []
for i in range(10):
    item = dataset[i]
    frame_starts.append(item['frame_start'])
print(f"Frame starts (limit=100): {frame_starts}")
for fs in frame_starts:
    assert fs <= args.train_frame_limit - args.num_frames

# Test 3: Validation set (no limit should be applied if we set is_train=False)
print("Testing no frame_start limit for validation...")
val_dataset = MDGenDataset(args, is_train=False)
frame_starts_val = []
for i in range(10):
    item = val_dataset[i]
    frame_starts_val.append(item['frame_start'])
print(f"Frame starts (no limit): {frame_starts_val}")
# Note: 1a62_A likely has more than 100 frames, so some might be > 50 (100-50=50)
has_high_frame = any(fs > (args.train_frame_limit - args.num_frames) for fs in frame_starts_val)
print(f"Any higher than limit? {has_high_frame}")

print("Verification script finished.")
