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
        self.args = args
        self.df = pd.read_csv(split, index_col='name')
        
        # Load the sequence mapping from atlas.csv
        # This maps '4o66_C' -> 'SEQUENCE...'
        if hasattr(args, 'atlas_csv'):
            seq_df = pd.read_csv(args.atlas_csv, index_col='name')
            self.seq_map = seq_df['seqres'].to_dict()
        else:
            raise ValueError("atlas_csv path not found in args. Please provide the sequence mapping file.")
        
        self.repeat = repeat
        self.num_consecutive = num_consecutive
        self.stride = stride
        
        self._build_frame_index()
    
    def _build_frame_index(self):
        self.frame_index = []
        # Total span of frames we need to extract per sample
        required_span = (self.num_consecutive - 1) * self.stride + 1
        
        for protein_idx, name in enumerate(self.df.index):
            # Extract folder name (4o66_C) from filename (4o66_C_R1)
            folder_name = name.split('_R')[0]
            
            # Resulting path: ./data/4o66_C/4o66_C_R1.npy
            npy_path = f'{self.args.data_dir}/{folder_name}/{name}{self.args.suffix}.npy'
            
            try:
                arr = np.lib.format.open_memmap(npy_path, 'r')
                num_frames = arr.shape[0]
                
                if self.args.frame_interval:
                    num_frames = len(range(0, num_frames, self.args.frame_interval))
                
                # Ensure we have enough room for the stride and consecutive count
                max_start_idx = num_frames - required_span
                
                if max_start_idx >= 0:
                    for frame_idx in range(max_start_idx + 1):
                        self.frame_index.append((protein_idx, frame_idx))
            except Exception as e:
                print(f"Warning: Could not load {npy_path}: {e}")
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
            full_name = name 
            folder_name = name.split('_R')[0]
            
            # Sequence lookup for overfit
            seqres = self.seq_map[folder_name]
            
            npy_path = f'{self.args.data_dir}/{folder_name}/{name}{self.args.suffix}.npy'
            arr = np.lib.format.open_memmap(npy_path, 'r')
            if self.args.frame_interval: 
                arr = arr[::self.args.frame_interval]
            
            required_span = (self.num_consecutive - 1) * self.stride + 1
            frame_start = np.random.choice(np.arange(arr.shape[0] - required_span + 1))
        else:
            protein_idx, frame_start = self.frame_index[idx]
            name = self.df.index[protein_idx]
            full_name = name 
            folder_name = name.split('_R')[0]
            
            # Sequence lookup from atlas_csv mapping
            try:
                seqres = self.seq_map[folder_name]
            except KeyError:
                raise KeyError(f"Protein {folder_name} not found in sequence mapping!")

            npy_path = f'{self.args.data_dir}/{folder_name}/{name}{self.args.suffix}.npy'
            arr = np.lib.format.open_memmap(npy_path, 'r')
            if self.args.frame_interval: 
                arr = arr[::self.args.frame_interval]

        # 2. Extract Strided Frames
        indices = [frame_start + i * self.stride for i in range(self.num_consecutive)]
        clean_frames = np.copy(arr[indices]).astype(np.float32) 

        # 3. Process Geometry
        seqres_encoded = np.array([restype_order[c] for c in seqres])
        aatype = torch.from_numpy(seqres_encoded)[None].expand(self.num_consecutive, -1)
        
        frames = atom14_to_frames(torch.from_numpy(clean_frames))
        atom37 = torch.from_numpy(atom14_to_atom37(clean_frames, aatype)).float()
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        
        L = clean_frames.shape[1]
        
        # 4. Spatial Cropping
        mask = np.ones(L, dtype=np.float32)
        if hasattr(self.args, 'crop_ratio') and self.args.crop_ratio < 1.0:
            seed_idx = np.random.randint(L)
            ref_coords = atom37[0, :, 1, :] 
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
            'mask': mask,
            'torsion_mask': torsion_mask[0], 
            'clean_trans': frames._trans,
            'clean_rots': frames._rots._rot_mats,
            'clean_torsions': torsions,
            'clean_atom37': atom37,
        }