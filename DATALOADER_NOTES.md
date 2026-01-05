# MDGenDataset Data Loading Notes

## 0. Data Preparation
Before loading data, ensure you have downloaded and preprocessed the necessary protein trajectories.
Use the provided helper script to automate this process:

```bash
# Usage: python scripts/download_and_prep.py <PROTEIN_ID>
python scripts/download_and_prep.py 1a62_A
```

This script will:
1. Download the raw data for `1a62_A` into `data/`.
2. Preprocess it into `.npy` format (e.g., `data/1a62_A_R1.npy`).

## 1. Protein Specification
You can specify which protein to load using the following arguments:

- **`--pep_name <ID>`**: 
  - If `<ID>` is found in the metadata CSV (default: `atlas.csv`), the sequence is looked up automatically.
  - If `<ID>` is **not** in the CSV, it is treated as a raw amino acid sequence (e.g., `AAAAA`).
- **`--pep_seq <STRING>`**: 
  - Explicitly provides the sequence string. This overrides any lookup or sequences inferred from `--pep_name`.
- **`--train_split <PATH>`**: 
  - Path to the CSV file used for lookups (e.g., `gen_model/splits/atlas.csv`). Required only if using Protein IDs.

## 2. Temporal Sampling & Frame Limits
- **`--train_frame_limit <N>`**: 
  - Restricts the training data to the first `N` frames of a trajectory.
  - **Important**: This only takes effect if `is_train=True` is passed to the `MDGenDataset` constructor.
- **Validation Sampling**: 
  - When `is_train=False`, the dataset samples across the **entire** trajectory, regardless of the limit. This allows you to monitor how the model generalizes beyond the training window.

## 3. Dataset Length & Sampling
- **The `repeat` Argument**: This is a constructor argument (default=1) that acts as a multiplier for the dataset length. Increasing `repeat` makes an "epoch" longer, which is useful for controlling validation and checkpoint frequency.
- **Single Peptide Mode**: 
  - If `pep_name` is set, the dataset returns a synthetic length of `1000 * repeat`.
  - **Why 1000?**: Since you are training on a single protein trajectory, a length of 1 would end the epoch immediately. A length of 1000 forces the `DataLoader` to pull 1000 random temporal snapshots (`frame_start`) from the simulation per epoch, providing the model with a variety of views of the protein's movement.

## 4. Output Structure
The `MDGenDataset` returns a dictionary for each sample with the following keys:

- **`name`**: (str) The identifier of the protein.
- **`frame_start`**: (int) The starting frame index for the random temporal crop.
- **`trans`**: (torch.Tensor) Cartesian coordinates (translations) of the backbone frames. Shape: `[num_frames, L, 3]`.
- **`rots`**: (torch.Tensor) Rotation matrices of the backbone frames. Shape: `[num_frames, L, 3, 3]`.
- **`torsions`**: (torch.Tensor) Torsion angles (dihedrals) in `sin`/`cos` representation. Shape: `[num_frames, L, 7, 2]`.
- **`torsion_mask`**: (torch.Tensor) Binary mask indicating which torsion angles are valid for each residue type. Shape: `[L, 7]`.
- **`seqres`**: (np.ndarray) Integer indices of the amino acid sequence. Shape: `[L]`.
- **`mask`**: (np.ndarray) General mask for valid residues/atoms (e.g., used for padding). Shape: `[L]`.

## 5. Usage Example (Python / Notebook)
```python
from gen_model.dataset import MDGenDataset

# Training Set (with ID and limit)
args.pep_name = "1a62_A"
args.train_frame_limit = 100
trainset = MDGenDataset(args, split="gen_model/splits/atlas.csv", is_train=True)

# Validation Set (full trajectory)
valset = MDGenDataset(args, split="gen_model/splits/atlas.csv", is_train=False)
```

## 5. Verification
A consolidated test suite is available to verify sequence mapping and frame sampling:
```bash
./env/bin/python tests/test_dataset.py
```
