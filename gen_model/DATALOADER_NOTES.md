# Dataset Loader Notes

## 1. SE(3) Invariance
Every frame is aligned to frame 0 via global superposition. This uses heavy atoms (protein but not hydrogen) to ensure a consistent coordinate system across the trajectory.

## 2. 4-Way Split Logic
Data is partitioned chronologically into four segments:
- train_early: Frames [0 to main training end]. Includes early and main data.
- train: Main training frames after the early threshold.
- val: Validation frames.
- test: Test frames.

Default early threshold: 5ns.
Default mode: 'train'.

## 3. Spatial Masking
Supports generative tasks by removing edge residues similar to image cropping. Only applied in 'train'/'train_early' when `crop_ratio` < 1.0.
- Goal: Uniform visibility probability ($k/L$) for all residues, removing core bias.
- Mechanism: Iterative Proportional Fitting (IPF) computes balanced seed weights.
- Reference Frame: Frame 0 (train_early) or First Train Frame (train).
- Selection: Weighted seed sampling -> N nearest **CA-CA neighbors** kept.
- Output: mask [L] tensor (1.0 = visible, 0.0 = masked).

## 4. Item Dictionary
- F: Number of consecutive frames (defined by num_consecutive)
- L: Sequence length (number of residues)
- B: Batch size

Each item (after default collation) contains:
- name: [1] List of protein identifiers (deduplicated).
- frame_indices: [B, F] indices used.
- seqres: [1, L] sequence indices (deduplicated).
- mask: [B, L] spatial mask (0/1).
- torsion_mask: [1, L, 7] chemical validity (deduplicated).
- clean_trans: [B, F, L, 3] CA positions.
- clean_rots: [B, F, L, 3, 3] rotation matrices.
- clean_torsions: [B, F, L, 7, 2] torsion sin/cos.
- clean_atom37: [B, F, L, 37, 3] all atom coordinates.

## 5. Filtering
To avoid batching errors, always filter to a single protein:
- --pep_name: Filter by protein (e.g., "4o66_C"). Default: None. 
- --replica: Filter by replica (e.g., "1" for _R1). Default: 1.
- --crop_ratio: Fraction of residues to keep (0.0-1.0). Default: 0.95

## 6. Sampling
- num_consecutive: Frames per sample. Changes tensor shape (F dimension). Default: 1.
- stride: Gap between consecutive frames in a sample. Default: 1.
- repeat: Oversampling factor. Changes epoch size, not data shape. Default: 1.

## 7. Basic Usage
dataset = MDGenDataset(args, mode='train')
# Ensure args.pep_name is set for batching.
