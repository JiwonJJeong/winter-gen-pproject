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
    def __init__(self, args, diffuser=None, split=None, mode='train', repeat=1, num_consecutive=1, stride=1):
        """
        Args:
            args: Global config object.
            diffuser: SE3Diffuser instance for applying diffusion noise.
            split: Path to the split CSV (optional).
            mode: Dataset mode ('train', 'val', 'test', 'train_early', or 'all').
            repeat: Oversampling factor.
            num_consecutive: Number of frames to return (1, 2, 3, etc.).
            stride: Gap between the consecutive frames (e.g., stride 2 picks frame 0, 2, 4).
        """
        super().__init__()
        self.args = _plain_args(args)
        self.mode = mode
        self._diffuser = diffuser
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
        
        self.balanced_weights = {}
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
            
            arr = np.lib.format.open_memmap(npy_path, 'r')
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

    def _apply_diffusion(self, clean_rigids, t, mask):
        """Apply SE(3) forward diffusion, or return zero-noise features when no diffuser."""
        L = mask.shape[0]
        if self._diffuser is not None:
            return self._diffuser.forward_marginal(
                rigids_0=clean_rigids, t=t, diffuse_mask=mask)
        return {
            'rigids_t':            clean_rigids.to_tensor_7(),
            'rot_score':           torch.zeros(L, 3),
            'trans_score':         torch.zeros(L, 3),
            'rot_score_scaling':   torch.tensor(1.0),
            'trans_score_scaling': torch.tensor(1.0),
        }

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
                arr = np.lib.format.open_memmap(npy_path, 'r')
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
        return len(self.frame_index)

    def __getitem__(self, idx):
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
        arr = np.lib.format.open_memmap(npy_path, 'r')
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

        # 5. Convert to Rigid format for SE3 diffusion
        clean_rigids = Rigid(
            rots=Rotation(rot_mats=frames._rots._rot_mats[frame_idx]),
            trans=frames._trans[frame_idx]
        )
        
        # 6. Sample diffusion time and apply SE(3) forward process.
        if self._diffuser is not None:
            t = (np.random.uniform(getattr(self.args, 'min_t', 0.01), 1.0)
                 if self._is_training else 1.0)
        else:
            t = 0.0
        diff_feats = self._apply_diffusion(clean_rigids, t, mask)
        
        # 7. Prepare atom representations using openfold transforms if available
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
        
        # 8. Build final feature dictionary for SE3 diffusion model
        output = {
            # Core identifiers
            'aatype': aatype_tensor,
            'seq_idx': torch.arange(1, L + 1),  # 1-indexed
            'chain_idx': torch.ones(L),  # Single chain
            'res_mask': torch.from_numpy(mask).float(),
            'fixed_mask': torch.zeros(L),  # No motifs (all residues diffused)
            
            # Ground truth structure (t=0)
            'rigids_0': clean_rigids.to_tensor_7(),
            'atom37_pos': atom37[frame_idx],
            'atom14_pos': atom14_pos,
            'torsion_angles_sin_cos': torsions[frame_idx],
            
            # Noised structure (at time t) - from diffuser
            **diff_feats,
            't': np.float32(t),  # float32 so DataLoader collates to Float32 tensor
            
            # Self-conditioning (initially zeros)
            'sc_ca_t': torch.zeros(L, 3),
            
            # Metadata
            'residx_atom14_to_atom37': residx_atom14_to_atom37,
            'atom37_mask': atom37_mask,
            'residue_index': torch.arange(L),
            
            # Original custom fields (for tracking/debugging)
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
    """Conditional SE(3) diffusion dataset with curriculum temporal gap.

    Each sample pairs a clean SOURCE frame with a noised TARGET frame that is
    k raw-trajectory strides away (k can be negative = going backward in time).

    The model receives:
      - sc_ca_t : CA positions of the clean source frame (centred + scaled)
      - rigids_t / rot_score / trans_score : noised TARGET frame at diffusion time t
      - k : signed integer stride count, added to input_feats for ϕ(k) embedding

    Curriculum learning (from the paper):
      - current_max_k starts at 1 (k ∈ {-1, +1})
      - grows up to max_k (default 3) as training progresses
      - updated externally by TemporalCurriculumCallback in train_conditional.py

    Frame index is built once for the full ±max_k range; curriculum only
    restricts which k values are drawn at sample time.
    """

    def __init__(self, *args, max_k: int = 3, current_max_k: int = 1, **kwargs):
        if max_k < 1:
            raise ValueError("ConditionalMDGenDataset requires max_k >= 1")
        if not (1 <= current_max_k <= max_k):
            raise ValueError(f"current_max_k must be in [1, {max_k}]")
        # Set these BEFORE super().__init__ so _build_frame_index can read them.
        self._max_k = max_k
        self.current_max_k = current_max_k
        # num_consecutive=1: source frame only; we handle the k-offset ourselves.
        kwargs['num_consecutive'] = 1
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Only need to override the frame-validity rule; everything else in
    # _build_frame_index is inherited unchanged from MDGenDataset.
    # ------------------------------------------------------------------
    def _add_frames_for_split(self, protein_idx, s, e):
        """Bidirectional: source must have max_k*stride frames on BOTH sides."""
        required_half = self._max_k * self.stride
        min_start = s + required_half
        max_start = e - required_half - 1
        if max_start >= min_start:
            for frame_idx in range(min_start, max_start + 1):
                self.frame_index.append((protein_idx, frame_idx))

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        # 1. Source frame via parent (single frame; frame_idx=0).
        #    output['atom14_pos'] and output['centroid'] are centred+scaled.
        output = super().__getitem__(idx)

        # 2. Sample k from curriculum range (excluding 0 — source ≠ target).
        k_vals = (list(range(-self.current_max_k, 0))
                  + list(range(1, self.current_max_k + 1)))
        k = int(np.random.choice(k_vals))

        # 3. Load target frame (k strides away from source).
        protein_idx, frame_start = self.frame_index[idx]
        name = self.df.index[protein_idx]
        folder_name = name.split('_R')[0]
        seqres = self.seq_map[folder_name]

        npy_path = f'{self.args.data_dir}/{folder_name}/{name}{self.args.suffix}.npy'
        arr = np.lib.format.open_memmap(npy_path, 'r')
        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]

        target_abs_idx = frame_start + k * self.stride
        target_atom14 = np.copy(arr[target_abs_idx]).astype(np.float32)[None]  # [1, L, 14, 3]

        # 4. Compute geometry for target frame.
        aatype_np = np.array([restype_order[c] for c in seqres])
        aatype_1 = torch.from_numpy(aatype_np)[None]  # [1, L]
        frames   = atom14_to_frames(torch.from_numpy(target_atom14))
        atom37   = torch.from_numpy(atom14_to_atom37(target_atom14, aatype_1)).float()
        torsions, _ = atom37_to_torsions(atom37, aatype_1)

        target_rigids = Rigid(
            rots=Rotation(rot_mats=frames._rots._rot_mats[0]),
            trans=frames._trans[0],
        )

        # 5. Apply SE(3) diffusion to the target with the same t and mask as source.
        mask_np = output['res_mask'].numpy()
        t = float(output['t'])
        diff_feats = self._apply_diffusion(target_rigids, t, mask_np)

        # 6. Centre and scale target_r0 with SOURCE centroid (for consistency with
        #    unconditional rigids_0).  rigids_t is left in Å — identical to the
        #    unconditional parent — so that trans_score (also in Å) stays consistent
        #    with the IPA's coordinate frame and the loss comparison is valid.
        centroid = output['centroid']  # [1, 3]
        target_r0 = target_rigids.to_tensor_7()
        target_r0[..., 4:] = (target_r0[..., 4:] - centroid) * self.coord_scale

        # 7. Patch output: all diffusion / torsion fields → target; sc_ca_t → source.
        output.update({
            **diff_feats,
            'rigids_0':               target_r0,
            'torsion_angles_sin_cos': torsions[0],
            'sc_ca_t':                output['atom14_pos'][:, 1, :],  # source CA [N,3]
            'k':                      torch.tensor(k, dtype=torch.long),
        })
        return output
