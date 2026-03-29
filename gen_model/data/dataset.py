import torch
import numpy as np
import pandas as pd
from gen_model.data.geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
from gen_model.data.residue_constants import restype_order, RESTYPE_ATOM37_MASK
from types import SimpleNamespace


def _plain_args(args):
    """Convert OmegaConf DictConfig to a SimpleNamespace at the boundary.

    The DictConfig stays in the caller (notebook); inside dataset code we use
    a plain object that accepts any value type without OmegaConf validation."""
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(args, DictConfig):
            return SimpleNamespace(**OmegaConf.to_container(args, resolve=True))
    except ImportError:
        pass
    return args


# SE3 diffusion
import numpy as np

from gen_model.utils.rigid_utils import Rigid, Rotation
try:
    from openfold.data import data_transforms
except ImportError:
    # Fallback if openfold not installed
    data_transforms = None

class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, split=None, mode='train', repeat=1, num_consecutive=1, stride=1,
                 virtual_epoch_size: int = 0):
        """
        Args:
            args: Global config object.
            split: Path to the split CSV (optional).
            mode: Dataset mode ('train', 'val', 'test', 'train_early', or 'all').
            repeat: Oversampling factor.
            num_consecutive: Number of frames to return (1, 2, 3, etc.).
            stride: Gap between the consecutive frames (e.g., stride 2 picks frame 0, 2, 4).
            virtual_epoch_size: If > 0, report this as __len__ and sample randomly
                (SinFusion-style: prevents overfitting on small single-trajectory datasets).
        """
        super().__init__()
        self.args = _plain_args(args)
        self.mode = mode
        self._is_training = mode in ['train', 'train_early']
        
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
        self._virtual_epoch_size = virtual_epoch_size

        self.balanced_weights = {}
        self._arr_cache: dict = {}   # path → np.ndarray; avoids repeated Drive reads
        self._build_frame_index()
        
        # Calculate or load coordination scale factor
        self.coord_scale = getattr(self.args, 'coord_scale', None)
        if self.coord_scale is None:
            if self.mode in ['train', 'train_early']:
                self.coord_scale = self._compute_coord_scale()
            else:
                self.coord_scale = 0.1

        if self.mode in ['train', 'train_early']:
            print(f"Dataset {mode} mode: coord_scale = {self.coord_scale:.4f}")

    def _compute_coord_scale(self):
        """Computes global scale factor from sampled training data."""
        print(f"Computing coordinate scale for {len(self.frame_index)} frames...")
        ca_coords = []
        
        # Sample frames to estimate statistics accurately but efficiently
        num_samples = min(500, len(self.frame_index))
        indices = np.random.choice(len(self.frame_index), num_samples, replace=False)
        
        for idx in indices:
            protein_idx, frame_start = self.frame_index[idx]
            name = self.df.index[protein_idx]
            folder_name = name.split('_R')[0]
            npy_path = f'{self.args.data_dir}/{folder_name}/{name}{self.args.suffix}.npy'
            
            arr = self._load_npy(npy_path)
            if self.args.frame_interval:
                # Simplified sampling logic for scale computation
                frame_data = arr[frame_start * self.args.frame_interval]
            else:
                frame_data = arr[frame_start]
            
            # Extract CA (index 1) and center it
            ca = frame_data[:, 1, :].astype(np.float32)
            ca = ca - np.mean(ca, axis=0, keepdims=True)
            ca_coords.append(ca)
        
        if not ca_coords:
            return 0.1
            
        all_ca = np.concatenate(ca_coords, axis=0) # [N_sampled * L, 3]
        std = np.std(all_ca)
        return float(1.0 / (std + 1e-8))

    def _load_npy(self, npy_path: str) -> np.ndarray:
        """Load a .npy trajectory array, caching it in memory to avoid Drive FUSE errors.

        Google Drive FUSE mounts can drop their connection between dataset
        iterations, causing open_memmap to fail with OSError 107 (Transport
        endpoint not connected).  Loading the whole array once and caching it
        in a dict removes the live file-handle dependency after the first read.
        """
        if npy_path not in self._arr_cache:
            self._arr_cache[npy_path] = np.load(npy_path)
        return self._arr_cache[npy_path]

    def _add_frames_for_split(self, protein_idx, s, e):
        """Add valid source-frame starts for split segment [s, e).

        Override in subclasses to change the validity criterion.
        Default: forward-only — requires (num_consecutive-1)*stride frames ahead.
        """
        required_span = (self.num_consecutive - 1) * self.stride + 1
        max_start = e - required_span
        if max_start >= s:
            for frame_idx in range(s, max_start + 1):
                self.frame_index.append((protein_idx, frame_idx))


    def _build_frame_index(self):
        self.frame_index = []
        required_span = (self.num_consecutive - 1) * self.stride + 1

        split_cols = ['train_early_end', 'train_end', 'val_end']
        has_splits = all(col in self.df.columns for col in split_cols)

        if not has_splits and self.mode != 'all':
            print(f"Warning: Split columns missing in split file. Using all frames regardless of mode='{self.mode}'.")

        for protein_idx, name in enumerate(self.df.index):
            folder_name = name.split('_R')[0]

            if getattr(self.args, 'pep_name', None) and folder_name != self.args.pep_name:
                continue
            if getattr(self.args, 'replica', None) and not name.endswith(f"_R{self.args.replica}"):
                continue

            npy_path = f'{self.args.data_dir}/{folder_name}/{name}{self.args.suffix}.npy'

            try:
                arr = self._load_npy(npy_path)
                num_frames = arr.shape[0]

                masking_enabled = self.mode not in ['val', 'test'] and getattr(self.args, 'crop_ratio', 0.95) < 1.0
                if folder_name not in self.balanced_weights and masking_enabled:
                    ref_frame_idx = 0
                    if has_splits and self.mode == 'train':
                        ref_frame_idx = int(self.df.loc[name, 'train_early_end'])
                        if self.args.frame_interval:
                            ref_frame_idx //= self.args.frame_interval
                        ref_frame_idx = min(ref_frame_idx, num_frames - required_span)
                    first_frame_ca = np.array(arr[ref_frame_idx, :, 1, :])
                    self.balanced_weights[folder_name] = self._compute_balanced_weights(
                        first_frame_ca, self.args.crop_ratio)

                if self.args.frame_interval:
                    num_frames = len(range(0, num_frames, self.args.frame_interval))

                if has_splits and self.mode != 'all':
                    t_early_e = int(self.df.loc[name, 'train_early_end'])
                    t_e       = int(self.df.loc[name, 'train_end'])
                    v_e       = int(self.df.loc[name, 'val_end'])
                    if self.args.frame_interval:
                        t_early_e //= self.args.frame_interval
                        t_e       //= self.args.frame_interval
                        v_e       //= self.args.frame_interval
                    if self.mode == 'train_early':
                        self._add_frames_for_split(protein_idx, 0, t_e)
                    elif self.mode == 'train':
                        self._add_frames_for_split(protein_idx, t_early_e, t_e)
                    elif self.mode == 'val':
                        self._add_frames_for_split(protein_idx, t_e, v_e)
                    elif self.mode == 'test':
                        self._add_frames_for_split(protein_idx, v_e, num_frames)
                else:
                    self._add_frames_for_split(protein_idx, 0, num_frames)

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
        if self._virtual_epoch_size > 0:
            return self._virtual_epoch_size
        return len(self.frame_index)

    def __getitem__(self, idx):
        # SinFusion-style virtual epoch: sample a random real index
        if self._virtual_epoch_size > 0:
            idx = np.random.randint(len(self.frame_index))
        # 1. Resolve Protein and Starting Frame
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
        arr = self._load_npy(npy_path)
        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]

        # 2. Extract Strided Frames (use first frame for single-frame training)
        indices = [frame_start + i * self.stride for i in range(self.num_consecutive)]
        clean_frames = np.copy(arr[indices]).astype(np.float32) 
        frame_idx = 0  # Use first frame for SE3 diffusion

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

        # 5. Convert to Rigid format and serialise as tensor_7 for batching
        clean_rigids = Rigid(
            rots=Rotation(rot_mats=frames._rots._rot_mats[frame_idx]),
            trans=frames._trans[frame_idx]
        )

        # 6. Prepare atom representations using openfold transforms if available
        aatype_tensor = aatype[frame_idx]
        atom14_pos = torch.from_numpy(clean_frames[frame_idx]).float()
        
        if data_transforms is not None:
            # Use openfold data transforms for proper atom14/atom37 handling
            chain_feats = {
                'aatype': aatype_tensor,
                'all_atom_positions': atom37[frame_idx].double(),
                'all_atom_mask': (atom37[frame_idx] != 0).any(dim=-1).double()
            }
            chain_feats = data_transforms.make_atom14_masks(chain_feats)
            chain_feats = data_transforms.make_atom14_positions(chain_feats)
            
            residx_atom14_to_atom37 = chain_feats['residx_atom14_to_atom37']
            atom37_mask = chain_feats['all_atom_mask']
            atom14_pos = chain_feats['atom14_gt_positions']
        else:
            # Fallback: use residue constants
            atom37_mask = torch.from_numpy(RESTYPE_ATOM37_MASK[aatype_tensor.numpy()])
            residx_atom14_to_atom37 = torch.zeros(L, 14, dtype=torch.long)
        
        # 7. Build feature dictionary — clean geometry only.
        #    Diffusion noise (rigids_t, rot_score, trans_score, t) is applied
        #    in the Lightning module's forward() following the SinFusion pattern.
        output = {
            # Core identifiers
            'aatype': aatype_tensor,
            'seq_idx': torch.arange(1, L + 1),  # 1-indexed
            'chain_idx': torch.ones(L),
            'res_mask': torch.from_numpy(mask).float(),
            'fixed_mask': torch.zeros(L),

            # Clean structure (t=0) — noise applied in forward()
            'rigids_0': clean_rigids.to_tensor_7(),
            'atom37_pos': atom37[frame_idx],
            'atom14_pos': atom14_pos,
            'torsion_angles_sin_cos': torsions[frame_idx],

            # Self-conditioning placeholder (source CA; overridden in conditional dataset)
            'sc_ca_t': torch.zeros(L, 3),

            # Metadata
            'residx_atom14_to_atom37': residx_atom14_to_atom37,
            'atom37_mask': atom37_mask,
            'residue_index': torch.arange(L),
            'name': full_name,
            'frame_indices': np.array(indices),
        }
        
        # 9. Apply Centering and Scaling to translations
        # Use CA atoms (index 1 in atom14) as the reference for centering
        ca_pos = output['atom14_pos'][:, 1, :] # [L, 3]
        centroid = torch.mean(ca_pos, dim=0, keepdim=True) # [1, 3]
        
        # Center and scale atom14
        output['atom14_pos'] = (output['atom14_pos'] - centroid.unsqueeze(0)) * self.coord_scale
        
        # Center and scale atom37
        output['atom37_pos'] = (output['atom37_pos'] - centroid.unsqueeze(0)) * self.coord_scale
        
        # Center and scale rigids translation
        # rigids_0 is [7] tensor (quaternion [4] + translation [3])
        output['rigids_0'][..., 4:] = (output['rigids_0'][..., 4:] - centroid) * self.coord_scale
        
        # Store normalization metadata
        output['centroid'] = centroid
        output['coord_scale'] = torch.tensor(self.coord_scale)
        
        return output



class ConditionalMDGenDataset(MDGenDataset):
    """STAR-MD L-frame window dataset (Diffusion Forcing training scheme).

    Each sample is a window of L consecutive MD frames at a common noise level
    τ ~ U[0, max_t] (applied in the Lightning module's forward(), not here).
    All L frames are denoising targets simultaneously; the block-causal mask in
    SpatioTemporalAttention provides the causal structure at training time.

    Physical time stride (delta_t) is sampled from LogUniform[0.01, 10] ns;
    the raw-frame stride k is derived from it and clamped to what the trajectory
    can support for L frames.  Frame direction is ±1 with equal probability
    (data augmentation); the window is sorted ascending before output.

    Returns (per sample, collated to [B, L, ...] by default PyTorch collation):
        rigids_0                [L, N, 7]    clean backbone frames
        torsion_angles_sin_cos  [L, N, 7, 2] torsion angles for all frames
        sc_ca_t                 [L, N, 3]    per-frame self-conditioning CA:
                                             sc_ca_t[0]=0, sc_ca_t[l]=CA of frame l-1
        frame_idx               [L]          absolute frame positions (for RoPE2D)
        delta_t                 scalar       actual physical stride in ns
        res_mask                [N]          spatial crop mask (same for all frames)
        aatype / seq_idx / fixed_mask / chain_idx  [N] or scalar
    """

    def __init__(self, *args, num_frames: int = 16,
                 ns_per_stored_frame: float = 0.1,
                 curriculum: bool = False, **kwargs):
        self._num_frames = num_frames
        self._ns_per_stored = ns_per_stored_frame
        self._curriculum = curriculum
        self._sample_counter = 0
        kwargs['num_consecutive'] = 1  # anchor frame only; window built in __getitem__
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Frame validity: anchor must have (L-1) frames on both sides at k=1
    # so any sampled window direction is feasible.
    # ------------------------------------------------------------------
    def _add_frames_for_split(self, protein_idx, s, e):
        half = self._num_frames - 1   # minimum frames required on each side
        min_start = s + half
        max_start = e - half - 1
        if max_start >= min_start:
            for fi in range(min_start, max_start + 1):
                self.frame_index.append((protein_idx, fi))

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        # 1. Anchor frame via parent — gives us centroid, res_mask, coord_scale.
        anchor = super().__getitem__(idx)

        protein_idx, frame_start = self.frame_index[idx]
        name        = self.df.index[protein_idx]
        folder_name = name.split('_R')[0]
        seqres      = self.seq_map[folder_name]

        npy_path = f'{self.args.data_dir}/{folder_name}/{name}{self.args.suffix}.npy'
        arr = self._load_npy(npy_path)
        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]
        total_frames = arr.shape[0]
        L = self._num_frames

        # 2. Sample physical stride and derive raw-frame stride k (Decision 4B).
        #    SinFusion curriculum: gradually increase delta_t range during training.
        #    Phase 1 (<75k samples): delta_t in [0.01, 0.1] ns (adjacent frames)
        #    Phase 2 (75k-150k):    delta_t in [0.01, 1.0] ns
        #    Phase 3 (>=150k):      delta_t in [0.01, 10.0] ns (full range)
        if self._curriculum:
            self._sample_counter += 1
            if self._sample_counter < 75_000:
                max_delta = 0.1
            elif self._sample_counter < 150_000:
                max_delta = 1.0
            else:
                max_delta = 10.0
        else:
            max_delta = 10.0
        delta_t_ns   = float(np.exp(np.random.uniform(np.log(0.01), np.log(max_delta))))
        k_raw        = max(1, int(round(delta_t_ns / self._ns_per_stored)))
        max_k_avail  = max(1, (total_frames - 1) // (L - 1))
        k            = min(k_raw, max_k_avail)
        delta_t_act  = k * self._ns_per_stored   # quantised to stored-frame grid

        # 3. Sample direction and build sorted window indices (Decision 3B).
        direction   = int(np.random.choice([-1, 1]))
        raw_indices = [frame_start + direction * i * k for i in range(L)]
        raw_indices = [max(0, min(total_frames - 1, fi)) for fi in raw_indices]
        frame_idx_arr = np.array(sorted(raw_indices), dtype=np.int64)   # [L]

        # 4. Load all L frames and compute geometry.
        frames_data = np.copy(arr[frame_idx_arr]).astype(np.float32)    # [L, N, 14, 3]
        N_res       = frames_data.shape[1]

        aatype_np = np.array([restype_order[c] for c in seqres])
        aatype_L  = torch.from_numpy(aatype_np)[None].expand(L, -1)     # [L, N]

        frames_rigid = atom14_to_frames(torch.from_numpy(frames_data))
        atom37_L     = torch.from_numpy(
            atom14_to_atom37(frames_data, aatype_L)).float()             # [L, N, 37, 3]
        torsions_L, _ = atom37_to_torsions(atom37_L, aatype_L)          # [L, N, 7, 2]

        # 5. Build rigids_0 for all L frames and apply centering + scaling.
        centroid    = anchor['centroid']     # [1, 3]
        scale       = self.coord_scale
        rigids_0_frames = []
        for l in range(L):
            r = Rigid(
                rots=Rotation(rot_mats=frames_rigid._rots._rot_mats[l]),
                trans=frames_rigid._trans[l],
            )
            r7 = r.to_tensor_7()
            r7[..., 4:] = (r7[..., 4:] - centroid) * scale
            rigids_0_frames.append(r7)
        rigids_0 = torch.stack(rigids_0_frames, dim=0)  # [L, N, 7]

        # 6. Per-frame self-conditioning CA.
        #    sc_ca_t[0] = zeros (no prior context); sc_ca_t[l] = CA of frame l-1.
        ca_all  = atom37_L[:, :, 1, :].clone()                          # [L, N, 3]
        ca_all  = (ca_all - centroid.unsqueeze(0)) * scale
        sc_ca_t = torch.zeros(L, N_res, 3)
        sc_ca_t[1:] = ca_all[:-1]

        return {
            'aatype':                   anchor['aatype'],
            'seq_idx':                  anchor['seq_idx'],
            'chain_idx':                anchor['chain_idx'],
            'res_mask':                 anchor['res_mask'],
            'fixed_mask':               anchor['fixed_mask'],
            # L-frame geometry (noise applied in Lightning module's forward())
            'rigids_0':                 rigids_0,                        # [L, N, 7]
            'torsion_angles_sin_cos':   torsions_L,                      # [L, N, 7, 2]
            'sc_ca_t':                  sc_ca_t,                         # [L, N, 3]
            # Temporal metadata for STAR-MD
            'frame_idx':                torch.tensor(frame_idx_arr, dtype=torch.long),  # [L]
            'delta_t':                  torch.tensor(delta_t_act, dtype=torch.float32),
            # Normalization metadata
            'centroid':                 centroid,
            'coord_scale':              anchor['coord_scale'],
            'name':                     anchor['name'],
        }
