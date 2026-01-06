# MIT License

# Copyright (c) 2024 Bowen Jing, Hannes Stärk, Tommi Jaakkola, Bonnie Berger

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from # https://github.com/bjing2016/mdgen/blob/master/mdgen/dataset.py

import torch
from .rigid_utils import Rigid
from .residue_constants import restype_order, RESTYPE_ATOM37_MASK
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
       
class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, repeat=1, k_steps=1):
        super().__init__()
        self.df = pd.read_csv(split, index_col='name')
        self.args = args
        self.repeat = repeat
        self.k_steps = k_steps
        
        # Build index mapping: dataset_idx -> (protein_idx, frame_idx)
        self._build_frame_index()
    
    def _build_frame_index(self):
        """Build an index that maps dataset indices to (protein_idx, frame_idx) pairs"""
        self.frame_index = []
        
        if self.args.overfit_peptide:
            # For overfitting, we'll handle this in __getitem__
            return
        
        for protein_idx, name in enumerate(self.df.index):
            # Determine the trajectory file to check
            if self.args.atlas:
                # For atlas, we'll use R1 to determine frame count
                # (assuming all replicates have the same number of frames)
                full_name = f"{name}_R1"
            else:
                full_name = name
            
            try:
                arr = np.lib.format.open_memmap(
                    f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r'
                )
                num_frames = arr.shape[0]
                
                if self.args.frame_interval:
                    num_frames = len(range(0, num_frames, self.args.frame_interval))
                
                # Calculate valid starting frames (need room for k_steps forward)
                max_start_idx = num_frames - self.k_steps - 1
                
                if max_start_idx >= 0:
                    # Add all valid frame indices for this protein
                    for frame_idx in range(max_start_idx + 1):
                        self.frame_index.append((protein_idx, frame_idx))
            except Exception as e:
                print(f"Warning: Could not load {full_name}: {e}")
                continue
        
        # Apply repeat factor
        if self.repeat > 1:
            self.frame_index = self.frame_index * self.repeat
    
    def __len__(self):
        if self.args.overfit_peptide:
            return 1000
        return len(self.frame_index)

    def __getitem__(self, idx):
        # Handle overfitting modes
        if self.args.overfit_peptide:
            # For overfit_peptide, use random frame selection
            name = self.args.overfit_peptide
            seqres = name
            protein_idx = 0
            
            # Load trajectory
            if self.args.atlas:
                i = np.random.randint(1, 4)
                full_name = f"{name}_R{i}"
            else:
                full_name = name
            
            arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
            if self.args.frame_interval:
                arr = arr[::self.args.frame_interval]
            
            # Random frame selection for overfitting
            max_start_idx = arr.shape[0] - self.k_steps - 1
            if max_start_idx < 0:
                raise ValueError(f"Trajectory too short: {arr.shape[0]} frames, need at least {self.k_steps + 1}")
            
            frame_idx = np.random.choice(np.arange(max_start_idx + 1))
            if self.args.overfit_frame:
                frame_idx = 0
        else:
            # Use deterministic frame indexing
            protein_idx, frame_idx = self.frame_index[idx]
            name = self.df.index[protein_idx]
            seqres = self.df.seqres[name]
            
            # Load trajectory
            if self.args.atlas:
                # Randomly select a replicate for atlas
                i = np.random.randint(1, 4)
                full_name = f"{name}_R{i}"
            else:
                full_name = name
            
            arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
            if self.args.frame_interval:
                arr = arr[::self.args.frame_interval]
            
            if self.args.overfit:
                # For overfit mode, always use frame 0
                frame_idx = 0
        
        # Get the initial frame and the frame k_steps forward
        frame_0 = np.copy(arr[frame_idx:frame_idx+1]).astype(np.float32)  # Shape: (1, L, 14, 3)
        frame_k = np.copy(arr[frame_idx + self.k_steps:frame_idx + self.k_steps + 1]).astype(np.float32)  # Shape: (1, L, 14, 3)
        
        if self.args.copy_frames:
            # If copy_frames is set, use the same frame for both
            frame_k = frame_0.copy()

        # Process frame 0 (initial frame)
        seqres = np.array([restype_order[c] for c in seqres])
        aatype = torch.from_numpy(seqres)[None]  # Shape: (1, L)
        
        # Frame 0 processing
        frames_0 = atom14_to_frames(torch.from_numpy(frame_0))
        atom37_0 = torch.from_numpy(atom14_to_atom37(frame_0, aatype)).float()
        
        # Frame k processing (with noise)
        # Add Gaussian noise to the atom positions
        noise_std = getattr(self.args, 'noise_std', 0.1)  # Default noise std of 0.1 Angstroms
        frame_k_noised = frame_k + np.random.normal(0, noise_std, frame_k.shape).astype(np.float32)
        
        frames_k = atom14_to_frames(torch.from_numpy(frame_k_noised))
        atom37_k = torch.from_numpy(atom14_to_atom37(frame_k_noised, aatype)).float()
        
        L = frames_0.shape[1]
        mask = np.ones(L, dtype=np.float32)
        
        if self.args.no_frames:
            return {
                'name': full_name,
                'frame_idx': frame_idx,
                'k_steps': self.k_steps,
                # Frame 0 data
                'atom37_0': atom37_0,
                # Frame k data (noised)
                'atom37_k': atom37_k,
                # Shared data
                'seqres': seqres,
                'mask': RESTYPE_ATOM37_MASK[seqres],
            }
        
        # Compute torsions for both frames
        torsions_0, torsion_mask_0 = atom37_to_torsions(atom37_0, aatype)
        torsions_k, torsion_mask_k = atom37_to_torsions(atom37_k, aatype)
        
        torsion_mask = torsion_mask_0[0]  # Use frame 0's torsion mask as reference
        


        # Masking logic
        if hasattr(self.args, 'crop_ratio') and self.args.crop_ratio > 0:
            ratio = self.args.crop_ratio
        else:
            ratio = 0.95 # Default crop ratio if not specified
        
        keep_len = int(L * ratio)
        if keep_len < L:
             seed_idx = np.random.randint(L)
             # Use CA atoms (index 1) of the first frame for distance calculation
             ref_coords = atom37_0[0, :, 1, :] # [L, 3]
             dists = torch.norm(ref_coords - ref_coords[seed_idx], dim=-1) # [L]
             
             _, keep_indices = torch.topk(dists, k=keep_len, largest=False)
             
             spatial_mask = torch.zeros(L, device=atom37_0.device)
             spatial_mask[keep_indices] = 1.0
             
             # Apply spatial mask only to residue-level mask
             # torsion_mask remains unchanged (reflects chemical validity only)
             mask = mask * spatial_mask.cpu().numpy()
        # Else (if keeping everything), mask stays as ones (from initialization)

        return {
            'name': full_name,
            'frame_idx': frame_idx,
            'k_steps': self.k_steps,
            # Frame 0 data (clean)
            'torsions_0': torsions_0,
            'trans_0': frames_0._trans,
            'rots_0': frames_0._rots._rot_mats,
            # Frame k data (noised)
            'torsions_k': torsions_k,
            'trans_k': frames_k._trans,
            'rots_k': frames_k._rots._rot_mats,
            # Shared data
            'torsion_mask': torsion_mask,
            'seqres': seqres,
            'mask': mask, # (L,)
        }