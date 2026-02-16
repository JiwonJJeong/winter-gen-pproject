# Dataset Integration Guide: MDGenDataset → SE(3) Diffusion Model

## Current State: What You Have

Your `MDGenDataset` (in `gen_model/dataset.py`) provides:

```python
{
    'name': str,                    # Protein identifier
    'frame_indices': np.array,      # Which frames were sampled
    'seqres': np.array,             # Amino acid sequence (encoded as integers)
    'mask': np.array,               # Spatial crop mask [L]
    'torsion_mask': torch.Tensor,   # Which torsions are valid [L, 7]
    'clean_trans': torch.Tensor,    # Translations [num_frames, L, 3]
    'clean_rots': torch.Tensor,     # Rotation matrices [num_frames, L, 3, 3]
    'clean_torsions': torch.Tensor, # Torsion angles [num_frames, L, 7, 2]
    'clean_atom37': torch.Tensor,   # All-atom coords [num_frames, L, 37, 3]
}
```

## Required: What SE(3) Diffusion Model Needs

Looking at `PdbDataset.__getitem__()` (lines 221-277), the model expects:

```python
{
    # Core identifiers
    'aatype': torch.Tensor,              # [L] amino acid types (0-20)
    'seq_idx': torch.Tensor,             # [L] residue indices (1, 2, 3, ...)
    'chain_idx': torch.Tensor,           # [L] chain identifiers
    'res_mask': torch.Tensor,            # [L] which residues exist (1.0 or 0.0)
    
    # Ground truth structure (clean, t=0)
    'rigids_0': torch.Tensor,            # [L, 7] rigid transforms (quat + trans)
    'atom37_pos': torch.Tensor,          # [L, 37, 3] all-atom positions
    'atom14_pos': torch.Tensor,          # [L, 14, 3] atom14 positions
    'torsion_angles_sin_cos': torch.Tensor,  # [L, 7, 2] torsion angles
    
    # Noised structure (at time t)
    'rigids_t': torch.Tensor,            # [L, 7] noised rigid transforms
    'rot_score': torch.Tensor,           # [L, 3] rotation score (ground truth)
    'trans_score': torch.Tensor,         # [L, 3] translation score (ground truth)
    'rot_score_scaling': float,          # Scaling factor for rotation score
    'trans_score_scaling': float,        # Scaling factor for translation score
    
    # Diffusion parameters
    't': float,                          # Time step in [0, 1]
    'fixed_mask': torch.Tensor,          # [L] which residues are fixed (motif)
    
    # Self-conditioning (optional)
    'sc_ca_t': torch.Tensor,             # [L, 3] previous predicted CA positions
    
    # Metadata
    'residx_atom14_to_atom37': torch.Tensor,  # [L, 14] mapping
    'atom37_mask': torch.Tensor,         # [L, 37] which atoms exist
    'residue_index': torch.Tensor,       # [L] original PDB residue numbers
}
```

---

## Missing Components: What You Need to Add

### 1. **Diffusion-Related Features** ⚠️ CRITICAL

Your dataset has **clean** structures but doesn't apply diffusion noise. You need to:

#### a) Add SE3Diffuser Integration

```python
from data_se3.data.se3_diffuser import SE3Diffuser

class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, diffuser, is_training=True, ...):
        # ... existing code ...
        self._diffuser = diffuser
        self._is_training = is_training
```

#### b) Apply Forward Diffusion in `__getitem__`

```python
def __getitem__(self, idx):
    # ... your existing code to get clean_trans, clean_rots ...
    
    # Convert to Rigid objects
    from openfold.utils.rigid_utils import Rigid, Rotation
    
    # Combine rotations and translations into Rigid
    clean_rigids = Rigid(
        rots=Rotation(rot_mats=clean_rots[0]),  # Use first frame
        trans=clean_trans[0]
    )
    
    # Sample time step
    if self._is_training:
        t = np.random.uniform(self.args.min_t, 1.0)
    else:
        t = 1.0  # Full noise for validation
    
    # Apply diffusion
    diff_feats = self._diffuser.forward_marginal(
        rigids_0=clean_rigids,
        t=t,
        diffuse_mask=mask  # Use your spatial mask
    )
    
    # diff_feats contains:
    # - 'rigids_t': noised rigids [L, 7]
    # - 'rot_score': rotation score [L, 3]
    # - 'trans_score': translation score [L, 3]
    # - 'rot_score_scaling': float
    # - 'trans_score_scaling': float
```

---

### 2. **Feature Format Conversions** 🔧

#### a) Rigid Representation

**What you have**: Separate `clean_trans` and `clean_rots`

**What's needed**: Combined `rigids_0` in 7D format `[quat(4) + trans(3)]`

```python
from openfold.utils.rigid_utils import Rigid, Rotation

# Convert rotation matrices to Rigid objects
clean_rigids = Rigid(
    rots=Rotation(rot_mats=clean_rots[0]),  # [L, 3, 3]
    trans=clean_trans[0]                     # [L, 3]
)

# Convert to 7D tensor format
rigids_0 = clean_rigids.to_tensor_7()  # [L, 7]
```

#### b) Amino Acid Types

**What you have**: `seqres` (encoded integers)

**What's needed**: `aatype` (same thing, just rename)

```python
feats['aatype'] = torch.from_numpy(seqres).long()
```

#### c) Sequence Indices

**What you have**: Nothing explicit

**What's needed**: `seq_idx` - residue position in chain (1, 2, 3, ...)

```python
L = len(seqres)
feats['seq_idx'] = torch.arange(1, L + 1)  # 1-indexed
```

#### d) Chain Indices

**What you have**: Nothing (single chain assumed)

**What's needed**: `chain_idx` - which chain each residue belongs to

```python
feats['chain_idx'] = torch.ones(L)  # All residues in chain 1
```

---

### 3. **Atom Representations** 🧬

#### a) Atom14 Positions

**What you have**: `clean_atom37` (37-atom representation)

**What's needed**: `atom14_pos` (14-atom representation)

You already have the conversion function! Just need to call it:

```python
# You already do this:
atom37 = torch.from_numpy(atom14_to_atom37(clean_frames, aatype))

# But you also need to keep atom14:
atom14_pos = torch.from_numpy(clean_frames)  # This IS atom14!
```

#### b) Atom Masks

**What's needed**: `atom37_mask` and `atom14_mask`

```python
# Atom37 mask: which atoms exist for each residue
atom37_mask = (atom37[0] != 0).any(dim=-1).float()  # [L, 37]

# Or use the constant from residue_constants
from gen_model.residue_constants import RESTYPE_ATOM37_MASK
atom37_mask = torch.from_numpy(RESTYPE_ATOM37_MASK[aatype[0]])
```

#### c) Atom14 to Atom37 Mapping

**What's needed**: `residx_atom14_to_atom37`

```python
from openfold.data import data_transforms

chain_feats = {
    'aatype': aatype[0],
    'all_atom_positions': atom37[0].double(),
    'all_atom_mask': atom37_mask.double()
}
chain_feats = data_transforms.make_atom14_masks(chain_feats)
residx_atom14_to_atom37 = chain_feats['residx_atom14_to_atom37']
```

---

### 4. **Metadata Fields** 📋

#### a) Residue Mask

**What you have**: `mask` (spatial crop mask)

**What's needed**: `res_mask` (which residues exist)

```python
feats['res_mask'] = torch.from_numpy(mask).float()
```

#### b) Fixed Mask

**What's needed**: `fixed_mask` (which residues are motifs/fixed)

For unconditional generation (no motifs):
```python
feats['fixed_mask'] = torch.zeros(L)  # All residues are diffused
```

For conditional generation with motifs:
```python
# Example: fix first 10 residues
fixed_mask = torch.zeros(L)
fixed_mask[:10] = 1.0
feats['fixed_mask'] = fixed_mask
```

#### c) Self-Conditioning

**What's needed**: `sc_ca_t` (previous prediction)

For first iteration or training without self-conditioning:
```python
feats['sc_ca_t'] = torch.zeros(L, 3)  # No self-conditioning
```

#### d) Residue Index

**What's needed**: `residue_index` (original PDB numbering)

```python
feats['residue_index'] = torch.arange(L)  # Or from your data if available
```

---

## Complete Integration Example

Here's how to modify your `__getitem__` method:

```python
def __getitem__(self, idx):
    # ... your existing code to load data ...
    
    # Extract first frame for single-frame training
    frame_idx = 0
    L = clean_frames.shape[1]
    
    # 1. Convert to Rigid format
    from openfold.utils.rigid_utils import Rigid, Rotation
    
    clean_rigids = Rigid(
        rots=Rotation(rot_mats=clean_rots[frame_idx]),
        trans=clean_trans[frame_idx]
    )
    
    # 2. Sample time and apply diffusion
    if self._is_training:
        t = np.random.uniform(self.args.min_t, 1.0)
        diff_feats = self._diffuser.forward_marginal(
            rigids_0=clean_rigids,
            t=t,
            diffuse_mask=mask
        )
    else:
        t = 1.0
        diff_feats = self._diffuser.sample_ref(
            n_samples=L,
            impute=clean_rigids,
            diffuse_mask=mask,
            as_tensor_7=True
        )
    
    # 3. Prepare atom representations
    from openfold.data import data_transforms
    
    aatype_tensor = torch.from_numpy(seqres_encoded).long()
    chain_feats = {
        'aatype': aatype_tensor,
        'all_atom_positions': clean_atom37[frame_idx].double(),
        'all_atom_mask': (clean_atom37[frame_idx] != 0).any(dim=-1).double()
    }
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    
    # 4. Build final feature dictionary
    return {
        # Identifiers
        'aatype': aatype_tensor,
        'seq_idx': torch.arange(1, L + 1),
        'chain_idx': torch.ones(L),
        'res_mask': torch.from_numpy(mask).float(),
        'fixed_mask': torch.zeros(L),  # No motifs
        
        # Ground truth (t=0)
        'rigids_0': clean_rigids.to_tensor_7(),
        'atom37_pos': clean_atom37[frame_idx],
        'atom14_pos': chain_feats['atom14_gt_positions'],
        'torsion_angles_sin_cos': torsions[frame_idx],
        
        # Noised (at time t)
        **diff_feats,  # Adds rigids_t, rot_score, trans_score, etc.
        't': t,
        
        # Self-conditioning
        'sc_ca_t': torch.zeros(L, 3),
        
        # Metadata
        'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
        'atom37_mask': chain_feats['all_atom_mask'],
        'residue_index': torch.arange(L),
        
        # Your custom fields (optional, for tracking)
        'name': full_name,
        'frame_indices': np.array([frame_idx]),
    }
```

---

## Initialization Changes

Update your dataset initialization to accept a diffuser:

```python
# In your training script
from data_se3.data.se3_diffuser import SE3Diffuser
from omegaconf import OmegaConf

# Load SE3 diffusion config
se3_conf = OmegaConf.create({
    'diffuse_rot': True,
    'diffuse_trans': True,
    'so3': {
        'schedule': 'logarithmic',
        'min_sigma': 0.1,
        'max_sigma': 1.5,
        'num_sigma': 1000,
        'cache_dir': './',
    },
    'r3': {
        'min_b': 0.1,
        'max_b': 20.0,
        'coordinate_scaling': 0.1,  # Scale Angstroms to ~1.0 range
    }
})

diffuser = SE3Diffuser(se3_conf)

# Create dataset with diffuser
dataset = MDGenDataset(
    args=args,
    diffuser=diffuser,
    is_training=True,
    split='train_split.csv',
    mode='train'
)
```

---

## Summary Checklist

- [ ] Add `diffuser` parameter to `__init__`
- [ ] Convert rotations + translations → `Rigid` objects
- [ ] Apply `diffuser.forward_marginal()` to get noised structures
- [ ] Add `seq_idx`, `chain_idx` fields
- [ ] Rename `seqres` → `aatype`
- [ ] Keep both `atom14_pos` and `atom37_pos`
- [ ] Add `atom37_mask`, `residx_atom14_to_atom37`
- [ ] Add `fixed_mask` (zeros for unconditional generation)
- [ ] Add `sc_ca_t` (zeros initially)
- [ ] Add `residue_index` metadata
- [ ] Return all diffusion outputs: `rigids_t`, `rot_score`, `trans_score`, etc.
- [ ] Add `t` (time step) to output

---

## Key Differences from PdbDataset

| Feature | PdbDataset | Your MDGenDataset |
|---------|-----------|-------------------|
| Data source | Preprocessed PDB files | MD trajectory .npy files |
| Frames | Single structure | Multiple consecutive frames |
| Diffusion | Applied in `__getitem__` | ❌ **Need to add** |
| Motif scaffolding | Supported via `fixed_mask` | ❌ **Need to add** |
| Self-conditioning | Optional | ❌ **Need to add** |
| Spatial cropping | Distance-based masking | ✅ Already have! |
| Multi-chain | Supported | Single chain (can extend) |

---

## Next Steps

1. **Add diffuser integration** (most critical)
2. **Test with a single sample** to ensure all tensor shapes match
3. **Verify with the training script** (`experiments/train_se3_diffusion.py`)
4. **Optional**: Add self-conditioning support
5. **Optional**: Add multi-chain support if needed

Let me know if you need help implementing any of these changes!
