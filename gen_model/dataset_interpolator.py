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
    def __init__(self, args, split, repeat=1):
        super().__init__()
        self.df = pd.read_csv(split, index_col='name')
        self.args = args
        self.repeat = repeat
    def __len__(self):
        if self.args.overfit_peptide:
            return 1000
        return self.repeat * len(self.df)

    def __getitem__(self, idx):
        idx = idx % len(self.df)
        if self.args.overfit:
            idx = 0

        # ... (Existing name/seqres loading logic) ...
        if self.args.overfit_peptide is None:
            name = self.df.index[idx]
            seqres = self.df.seqres[name]
        else:
            name = self.args.overfit_peptide
            seqres = name
            
        # ... (Existing memmap loading logic) ...
        if self.args.atlas:
             i = np.random.randint(1, 4)
             full_name = f"{name}_R{i}"
        else:
             full_name = name
        
        arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]
        
        # --- MODIFICATION START: Triple Frame Selection ---
        
        # 1. Select a start frame such that we can grab exactly 3 frames
        # Ensure we don't go out of bounds
        max_start = arr.shape[0] - 3
        if max_start < 0:
             # Handle edge case where video is too short
             raise ValueError(f"Trajectory {full_name} is too short ({arr.shape[0]}) for a 3-frame window.")
             
        frame_start = np.random.choice(np.arange(max_start))
        if self.args.overfit_frame:
            frame_start = 0
            
        # 2. Extract the Ground Truth (Clean) Triplet
        # Shape: (3, Atoms, 3)
        clean_arr = np.copy(arr[frame_start : frame_start + 3]).astype(np.float32)
        
        # 3. Create the Input (Noised) Triplet
        input_arr = clean_arr.copy()
        
        # Define noise scale (use arg or default to 0.1 Angstroms)
        noise_scale = getattr(self.args, 'noise_scale', 0.1)
        
        # Apply Gaussian noise ONLY to the middle frame (index 1)
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=input_arr[1].shape).astype(np.float32)
        input_arr[1] += noise
        
        # --- MODIFICATION END ---

        # 4. Compute Geometry Features based on the NOISED Input
        # We use input_arr to generate the features the model sees
        frames = atom14_to_frames(torch.from_numpy(input_arr))
        seqres = np.array([restype_order[c] for c in seqres])
        aatype = torch.from_numpy(seqres)[None].expand(3, -1) # Expand to 3 frames
        
        # Convert Noised Atom14 -> Noised Atom37
        atom37 = torch.from_numpy(atom14_to_atom37(input_arr, aatype)).float()
        
        L = frames.shape[1]
        mask = np.ones(L, dtype=np.float32)
        
        # Torsions calculated from the Noised structure
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        torsion_mask = torsion_mask[0]

        # 5. Spatial Cropping Logic 
        # (Uses Frame 0 which is CLEAN, so the crop region is stable)
        if hasattr(self.args, 'crop_ratio') and self.args.crop_ratio > 0:
            ratio = self.args.crop_ratio
        else:
            ratio = 0.95
        
        keep_len = int(L * ratio)
        if keep_len < L:
             seed_idx = np.random.randint(L)
             # Note: We use atom37[0], which corresponds to input_arr[0] (CLEAN)
             ref_coords = atom37[0, :, 1, :] 
             dists = torch.norm(ref_coords - ref_coords[seed_idx], dim=-1)
             _, keep_indices = torch.topk(dists, k=keep_len, largest=False)
             spatial_mask = torch.zeros(L, device=atom37.device)
             spatial_mask[keep_indices] = 1.0
             mask = mask * spatial_mask.cpu().numpy()

        return {
            'name': full_name,
            'frame_start': frame_start,
            # Inputs (Noised middle frame features)
            'torsions': torsions,          # Shape: (3, L, 7, 2)
            'torsion_mask': torsion_mask,
            'trans': frames._trans,        # Shape: (3, L, 3)
            'rots': frames._rots._rot_mats, # Shape: (3, L, 3, 3)
            
            # Metadata
            'seqres': seqres,
            'mask': mask,
            
            # Supervision Target
            # We return the raw coordinates of the clean middle frame
            # Shape: (Atoms, 3) or (14, 3) depending on your atom14 convention
            'clean_coords': torch.from_numpy(clean_arr[1]).float(),
            'clean_atom37': torch.from_numpy(atom14_to_atom37(clean_arr, aatype)[1]).float()
        }