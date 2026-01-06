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

        if self.args.overfit_peptide is None:
            name = self.df.index[idx]
            seqres = self.df.seqres[name]
        else:
            name = self.args.overfit_peptide
            seqres = name

        if self.args.atlas:
            i = np.random.randint(1, 4)
            full_name = f"{name}_R{i}"
        else:
            full_name = name
            
        arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]
        
        # --- 1. Select Single Random Frame ---
        frame_idx = np.random.choice(np.arange(arr.shape[0]))
        if self.args.overfit_frame:
            frame_idx = 0
            
        # Load Clean Data and reshape to (1, N_atoms, 3) to maintain batch dims
        clean_arr = np.copy(arr[frame_idx])[None, ...].astype(np.float32)
        
        # --- 2. Inject Noise ---
        # Create input array (Noised)
        input_arr = clean_arr.copy()
        
        # Get noise scale from args or default to 0.1
        noise_scale = getattr(self.args, 'noise_scale', 0.1)
        
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=input_arr.shape).astype(np.float32)
        input_arr += noise

        # --- 3. Compute Features on NOISED Data ---
        # The model sees the distorted geometry
        frames = atom14_to_frames(torch.from_numpy(input_arr))
        
        seqres = np.array([restype_order[c] for c in seqres])
        # Expand seqres to match the single frame dimension (1, L)
        aatype = torch.from_numpy(seqres)[None].expand(1, -1)
        
        atom37 = torch.from_numpy(atom14_to_atom37(input_arr, aatype)).float()
        
        L = frames.shape[1]
        mask = np.ones(L, dtype=np.float32)
        
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        torsion_mask = torsion_mask[0]

        # --- 4. Spatial Cropping (GNN Subgraph) ---
        if hasattr(self.args, 'crop_ratio') and self.args.crop_ratio > 0:
            ratio = self.args.crop_ratio
        else:
            ratio = 0.95 # Default crop ratio
        
        keep_len = int(L * ratio)
        if keep_len < L:
             seed_idx = np.random.randint(L)
             
             # Calculate distance based on the NOISED input
             # This ensures the crop is consistent with what the model "sees"
             ref_coords = atom37[0, :, 1, :] # [L, 3] (CA atoms)
             dists = torch.norm(ref_coords - ref_coords[seed_idx], dim=-1) # [L]
             
             _, keep_indices = torch.topk(dists, k=keep_len, largest=False)
             
             spatial_mask = torch.zeros(L, device=atom37.device)
             spatial_mask[keep_indices] = 1.0
             
             mask = mask * spatial_mask.cpu().numpy()

        return {
            'name': full_name,
            'frame_idx': frame_idx,
            
            # Input Features (Noised)
            'torsions': torsions,           # Shape: (1, L, 7, 2)
            'torsion_mask': torsion_mask,
            'trans': frames._trans,         # Shape: (1, L, 3)
            'rots': frames._rots._rot_mats, # Shape: (1, L, 3, 3)
            'seqres': seqres,
            'mask': mask,                   # (L,)
            
            # Ground Truth Targets (Clean)
            # Use this for loss calculation: MSE(model_pred, clean_coords)
            'clean_coords': torch.from_numpy(clean_arr).float(), # Shape: (1, 14, 3)
            'clean_atom37': torch.from_numpy(atom14_to_atom37(clean_arr, aatype)).float()
        }