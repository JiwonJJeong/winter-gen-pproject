# Dataset Loader Documentation

## Overview

The dataset loaders (`dataset_projector.py`, `dataset_predictor.py`, `dataset_interpolator.py`) load molecular dynamics (MD) trajectory data and apply **spatial masking** to remove 5% of residues based on geometric proximity.

## Key Features

### 1. Spatial Masking (Induced Subgraph Sampling)
- **Removes 5% of residues** based on 3D spatial distance from a random seed
- **Preserves full tensor shapes** - no slicing occurs
- **Separates concerns**: `mask` controls spatial visibility, `torsion_mask` reflects chemical validity

### 2. Data Returned

Each `__getitem__` call returns a dictionary with:

```python
{
    'name': str,              # Protein name
    'frame_start': int,       # Starting frame index
    'torsions': Tensor,       # [Frames, L, 7, 2] - sin/cos of 7 torsion angles
    'torsion_mask': Tensor,   # [L, 7] - which torsions are chemically valid
    'trans': Tensor,          # [Frames, L, 3] - CA translations
    'rots': Tensor,           # [Frames, L, 3, 3] - rotation matrices
    'seqres': ndarray,        # [L] - amino acid type indices
    'mask': ndarray,          # [L] - which residues are spatially visible
}
```

### 3. Mask Semantics

**`mask` (shape `[L]`)**:
- `1.0` = residue is spatially visible (kept in 95%)
- `0.0` = residue is spatially masked (removed 5%)
- Controls which residues the model should use

**`torsion_mask` (shape `[L, 7]`)**:
- `1.0` = torsion angle exists for this amino acid
- `0.0` = torsion angle doesn't exist (e.g., Glycine has no chi angles)
- **NOT affected by spatial masking** - only reflects chemical validity

### 4. No Data Zeroing

The actual values in `torsions`, `trans`, and `rots` are **preserved** for all residues, including masked ones. Only the `mask` indicates which to ignore.

## Spatial Masking Algorithm

```python
# 1. Select random seed residue
seed_idx = np.random.randint(L)

# 2. Compute distances from seed (using CA atoms of first frame)
ref_coords = atom37[0, :, 1, :]  # [L, 3]
dists = torch.norm(ref_coords - ref_coords[seed_idx], dim=-1)

# 3. Keep nearest 95% of residues
keep_len = int(L * 0.95)
_, keep_indices = torch.topk(dists, k=keep_len, largest=False)

# 4. Create spatial mask
spatial_mask = torch.zeros(L)
spatial_mask[keep_indices] = 1.0

# 5. Apply to residue-level mask only
mask = mask * spatial_mask.cpu().numpy()
# torsion_mask is NOT modified
```

## Model Usage

Your model should check the `mask` to ignore spatially masked residues:

```python
# Simple approach: multiply losses by mask
loss = criterion(predictions, targets)
masked_loss = loss * mask  # Zero out loss for masked residues
final_loss = masked_loss.sum() / mask.sum()

# With torsion_mask for per-torsion losses
loss = criterion(predictions, targets)  # [L, 7]
loss = loss * mask.unsqueeze(-1) * torsion_mask
final_loss = loss.sum() / (mask.sum() * torsion_mask.sum())
```

## Testing

Run comprehensive tests:
```bash
pytest tests/test_dataset.py -v
```

Tests verify:
- Exactly 5% removal
- Shape preservation
- Torsion mask independence
- No data zeroing
- Randomness across samples
- All dataset variants work correctly

## Configuration

Set `crop_ratio` in your args:
```python
args.crop_ratio = 0.95  # Keep 95%, remove 5%
```

Default is `0.95` if not specified.

## Data Preparation

Before loading data, download and preprocess protein trajectories:

```bash
python scripts/download_and_prep.py 1a62_A
```

This downloads raw data and preprocesses it into `.npy` format.

## Implementation Notes

### Dependencies Fixed
- Removed `dm-tree` dependency from `residue_constants.py`
- Added numpy support to `batched_gather` in `tensor_utils.py`

### Dataset Variants
All three variants implement identical spatial masking:
- `dataset_projector.py` - for projection tasks
- `dataset_predictor.py` - for prediction tasks  
- `dataset_interpolator.py` - for interpolation tasks
