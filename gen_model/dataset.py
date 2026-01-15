import torch
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
from .residue_constants import restype_order, RESTYPE_ATOM37_MASK

class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, split=None, mode='train', repeat=1, num_consecutive=1, stride=1):
        """
        Args:
            args: Global config object.
            split: Path to the split CSV (optional).
            mode: Dataset mode ('train', 'val', 'test', 'train_early', or 'all').
            repeat: Oversampling factor.
            num_consecutive: Number of frames to return (1, 2, 3, etc.).
            stride: Gap between the consecutive frames (e.g., stride 2 picks frame 0, 2, 4).
        """
        super().__init__()
        self.args = args
        self.mode = mode
        
        # Determine split file if not provided
        if split is None:
            # Check for overrides in args, otherwise use the master split file
            if mode in ['train', 'train_early']:
                split = getattr(args, 'train_split', None)
            elif mode == 'val':
                split = getattr(args, 'val_split', None)
            
            split = split or 'gen_model/splits/frame_splits.csv'
        
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
        
        self.balanced_weights = {}
        self._build_frame_index()
    
    def _build_frame_index(self):
        self.frame_index = []
        # Total span of frames we need to extract per sample
        required_span = (self.num_consecutive - 1) * self.stride + 1
        
        split_cols = ['train_early_end', 'train_end', 'val_end']
        has_splits = all(col in self.df.columns for col in split_cols)
        
        if not has_splits and self.mode != 'all':
             print(f"Warning: Split columns missing in split file. Using all frames regardless of mode='{self.mode}'.")

        for protein_idx, name in enumerate(self.df.index):
            folder_name = name.split('_R')[0]
            
            # Filter by pep_name if provided
            if getattr(self.args, 'pep_name', None) and folder_name != self.args.pep_name:
                continue
            
            # Filter by replica if provided
            if getattr(self.args, 'replica', None) and not name.endswith(f"_R{self.args.replica}"):
                continue
                
            npy_path = f'{self.args.data_dir}/{folder_name}/{name}{self.args.suffix}.npy'
            
            try:
                arr = np.lib.format.open_memmap(npy_path, 'r')
                num_frames = arr.shape[0]

                # Determine reference frame for IPF if masking is needed
                masking_enabled = self.mode not in ['val', 'test'] and getattr(self.args, 'crop_ratio', 0.95) < 1.0
                if folder_name not in self.balanced_weights and masking_enabled:
                    ref_frame_idx = 0
                    if has_splits and self.mode == 'train':
                        ref_frame_idx = int(self.df.loc[name, 'train_early_end'])
                        if self.args.frame_interval:
                            ref_frame_idx //= self.args.frame_interval
                        # Clamp to valid range
                        ref_frame_idx = min(ref_frame_idx, num_frames - required_span)
                    
                    first_frame_ca = np.array(arr[ref_frame_idx, :, 1, :])
                    self.balanced_weights[folder_name] = self._compute_balanced_weights(first_frame_ca, self.args.crop_ratio)
                
                if self.args.frame_interval:
                    num_frames = len(range(0, num_frames, self.args.frame_interval))
                
                def add_valid_frames(s, e):
                    max_start_idx = e - required_span
                    if max_start_idx >= s:
                        for frame_idx in range(s, max_start_idx + 1):
                            self.frame_index.append((protein_idx, frame_idx))

                if has_splits and self.mode != 'all':
                    t_early_e = int(self.df.loc[name, 'train_early_end'])
                    t_e = int(self.df.loc[name, 'train_end'])
                    v_e = int(self.df.loc[name, 'val_end'])
                    
                    if self.args.frame_interval:
                        t_early_e //= self.args.frame_interval
                        t_e //= self.args.frame_interval
                        v_e //= self.args.frame_interval

                    if self.mode == 'train_early':
                        # Early + Train (from instruction: randomly return from both)
                        add_valid_frames(0, t_e)
                    elif self.mode == 'train':
                        add_valid_frames(t_early_e, t_e)
                    elif self.mode == 'val':
                        add_valid_frames(t_e, v_e)
                    elif self.mode == 'test':
                        add_valid_frames(v_e, num_frames)
                else:
                    add_valid_frames(0, num_frames)

            except Exception as e:
                print(f"Warning: Could not load {npy_path}: {e}")
                continue
        
        if self.repeat > 1:
            self.frame_index = self.frame_index * self.repeat

    def _compute_balanced_weights(self, coords, crop_ratio):
        """
        Solves for seed selection weights that result in uniform inclusion probability
        for all residues using Iterative Proportional Fitting (IPF).
        """
        L = len(coords)
        if L == 0: return np.array([])
        k = max(1, int(L * crop_ratio))
        
        # Compute neighbor matrix M
        coords_torch = torch.from_numpy(coords).float()
        dist_sq = torch.sum((coords_torch.unsqueeze(1) - coords_torch.unsqueeze(0))**2, dim=-1)
        _, top_indices = torch.topk(dist_sq, k=k, largest=False, dim=1) # Neighbors for each seed i
        
        M = torch.zeros((L, L))
        M.scatter_(1, top_indices, 1.0) # M[i, j] = 1 if j is in top-k of i
        
        target = k / L
        w = torch.ones(L) / L
        
        # IPF iterations
        for _ in range(50):
            current_prob = torch.mv(M.t(), w) # P_j = sum_i w_i M_ij
            correction = target / (current_prob + 1e-10)
            w = w * torch.mv(M, correction) / k
            w /= (w.sum() + 1e-10)
            
        return w.numpy()
    
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
        
        # 4. Spatial Cropping (only for training modes)
        mask = np.ones(L, dtype=np.float32)
        masking_enabled = self.mode not in ['val', 'test'] and getattr(self.args, 'crop_ratio', 0.95) < 1.0
        
        if masking_enabled:
            if folder_name in self.balanced_weights:
                weights = self.balanced_weights[folder_name]
                seed_idx = np.random.choice(L, p=weights)
            else:
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