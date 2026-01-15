# Dataset Loader Documentation

## Overview

The `MDGenDataset` (in `gen_model/dataset.py`) loads molecular dynamics (MD) trajectory data from ATLAS. It handles data partitioning (training/validation/test), geometric processing, and **spatial masking**.

## 1. SE(3) Invariance

SE(3) invariance (independence from global rotation and translation) is achieved through **global superposition** during the preprocessing stage:

- **Alignment**: Every frame in the trajectory is aligned to the first frame (`frame 0`) using a least-squares fit.
- **Selection**: The alignment uses all heavy atoms (`protein and not name H*`).
- **Consistency**: This ensures that all frames within a `.npy` file are in a shared coordinate system, allowing the model to focus on internal structural changes.

## 2. 4-Way Contiguous Splitting

The dataset supports a 4-way split logic based on simulation time. Simulations are partitioned into four contiguous segments:

| Mode | Range | Description |
| :--- | :--- | :--- |
| `train_early` | `[0, train_end)` | **Includes both** early and main training data. |
| `train` | `[early_end, train_end)` | Main training segment (after the early part). |
| `val` | `[train_end, val_end)` | Validation segment. |
| `test` | `[val_end, total_frames)` | Test segment. |

- `early_end` is determined by a nanosecond threshold (default 5ns).
- `train`, `val`, and `test` segments are partitioned by configurable ratios.
- The `MDGenDataset` defaults to `mode='train'`.

## 3. Spatial Masking (Induced Subgraph Sampling)

The dataloader implements a spatial masking technique to support generative tasks:
- **Masking Percentage**: Controlled by `args.crop_ratio` (default `0.95`, i.e., 5% masked).
- **Mechanism**: A random seed residue is chosen, and the nearest residues (by CA distance in the first frame) are kept.
- **`mask` tensor**: A `[L]` tensor where `1.0` is visible and `0.0` is masked.
- **`torsion_mask`**: Reflects chemical validity (e.g., presence of chi angles) and is **not** affected by spatial masking.

## 4. Item Dictionary

Each `__getitem__` call returns:

```python
{
    'name': str,              # Protein name (e.g., 4o66_C_R1)
    'frame_indices': ndarray, # Indices of frames returned
    'seqres': ndarray,        # Amino acid type indices
    'mask': ndarray,          # Spatial mask [L]
    'torsion_mask': Tensor,   # Chemical validity mask [L, 7]
    'clean_trans': Tensor,    # CA translations [F, L, 3]
    'clean_rots': Tensor,     # Rotation matrices [F, L, 3, 3]
    'clean_torsions': Tensor, # Torsion sin/cos [F, L, 7, 2]
    'clean_atom37': Tensor,   # Full atom coordinates [F, L, 37, 3]
}
```

## 6. Protein and Replica Filtering

To ensure consistent sequence lengths within a batch (avoiding `stack` errors), it is recommended to filter the dataset to a single protein. You can also filter by simulation replica.

- **`--pep_name`**: Filters trajectories to only include the specified protein name (e.g., `4o66_C`).
    - **Default**: `None` (loads all proteins in the split file).
    - **Note**: Batching proteins of different lengths will cause a `RuntimeError`.
- **`--replica`**: Filters trajectories to only include the specified replica number (e.g., `1` for `_R1`).
    - **Default**: `1` (loads the first replica only).
    - **Note**: Set to `None` in `args` (or avoid passing it if using a custom script) to load all replicas.

## 7. Usage

### Preprocessing
To download, preprocess, and split a new protein:
```bash
python scripts/download_and_prep.py 1a62_A --train_early_ns 50.0 --ratios 0.6 0.2 0.2
```

### Loading
```python
# The 'split' argument is completely optional.
# It defaults to 'gen_model/splits/frame_splits.csv'
# Ensure args.pep_name is set to avoid mixed-length batches.
args.pep_name = "4o66_C"
args.replica = 1 # Default
dataset = MDGenDataset(args, mode='train')
```
