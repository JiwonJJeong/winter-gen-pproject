import torch
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
from .residue_constants import restype_order, RESTYPE_ATOM37_MASK

class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, repeat=1, num_consecutive=1, stride=1):
        """
        Args:
            args: Global config object.
            split: Path to the split CSV.
            repeat: Oversampling factor.
            num_consecutive: Number of frames to return (1, 2, 3, etc.).
            stride: Gap between the consecutive frames (e.g., stride 2 picks frame 0, 2, 4).
        """
        super().__init__()
        self.df = pd.read_csv(split, index_col='name')
        self.args = args
        self.repeat = repeat
        self.num_consecutive = num_consecutive
        self.stride = stride
        
        # Build index mapping: dataset_idx -> (protein_idx, frame_start)
        self._build_frame_index()
    
    def _build_frame_index(self):
        self.frame_index = []
        # Total span of frames we need to extract per sample
        required_span = (self.num_consecutive - 1) * self.stride + 1
        
        for protein_idx, name in enumerate(self.df.index):
            full_name = f"{name}_R1" if self.args.atlas else name
            
            try:
                arr = np.lib.format.open_memmap(
                    f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r'
                )
                num_frames = arr.shape[0]
                
                if self.args.frame_interval:
                    num_frames = len(range(0, num_frames, self.args.frame_interval))
                
                # Ensure we have enough room for the stride and consecutive count
                max_start_idx = num_frames - required_span
                
                if max_start_idx >= 0:
                    for frame_idx in range(max_start_idx + 1):
                        self.frame_index.append((protein_idx, frame_idx))
            except Exception as e:
                print(f"Warning: Could not load {full_name}: {e}")
                continue
        
        if self.repeat > 1:
            self.frame_index = self.frame_index * self.repeat
    
    def __len__(self):
        if self.args.overfit_peptide: return 1000
        return len(self.frame_index)

    def __getitem__(self, idx):
        # 1. Resolve Protein and Starting Frame
        if self.args.overfit_peptide:
            name = self.args.overfit_peptide
            seqres = name # Assuming name is the sequence for overfitting
            protein_idx = 0
            # Reload for shape info
            full_name = f"{name}_R1" if self.args.atlas else name
            arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
            if self.args.frame_interval: arr = arr[::self.args.frame_interval]
            
            required_span = (self.num_consecutive - 1) * self.stride + 1
            frame_start = np.random.choice(np.arange(arr.shape[0] - required_span + 1))
        else:
            protein_idx, frame_start = self.frame_index[idx]
            name = self.df.index[protein_idx]
            seqres = self.df.seqres[name]
            full_name = f"{name}_R{np.random.randint(1,4)}" if self.args.atlas else name
            arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
            if self.args.frame_interval: arr = arr[::self.args.frame_interval]

        # 2. Extract Strided Frames
        # We pick frames: [start, start + stride, start + 2*stride, ...]
        indices = [frame_start + i * self.stride for i in range(self.num_consecutive)]
        clean_frames = np.copy(arr[indices]).astype(np.float32) # Shape: (num_consecutive, L, 14, 3)

        # 3. Process Geometry (ALL CLEAN)
        seqres_encoded = np.array([restype_order[c] for c in seqres])
        aatype = torch.from_numpy(seqres_encoded)[None].expand(self.num_consecutive, -1)
        
        # Convert to Rigid Frames and Torsions
        frames = atom14_to_frames(torch.from_numpy(clean_frames))
        atom37 = torch.from_numpy(atom14_to_atom37(clean_frames, aatype)).float()
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        
        L = clean_frames.shape[1]
        
        # 4. Spatial Cropping (95% Induced Subgraph)
        mask = np.ones(L, dtype=np.float32)
        if hasattr(self.args, 'crop_ratio') and self.args.crop_ratio < 1.0:
            seed_idx = np.random.randint(L)
            ref_coords = atom37[0, :, 1, :] # Use first frame CA for distance
            dists = torch.norm(ref_coords - ref_coords[seed_idx], dim=-1)
            keep_len = int(L * self.args.crop_ratio)
            _, keep_indices = torch.topk(dists, k=keep_len, largest=False)
            
            spatial_mask = torch.zeros(L)
            spatial_mask[keep_indices] = 1.0
            mask = mask * spatial_mask.numpy()

        # 5. Output Dictionary
        return {
            'name': full_name,
            'frame_indices': np.array(indices),
            'seqres': seqres_encoded,
            'mask': mask,               # (L,)
            'torsion_mask': torsion_mask[0], 
            # Clean Geometry Data
            'clean_trans': frames._trans,         # (N, L, 3)
            'clean_rots': frames._rots._rot_mats, # (N, L, 3, 3)
            'clean_torsions': torsions,           # (N, L, 7, 2)
            'clean_atom37': atom37,               # (N, L, 37, 3)
        }