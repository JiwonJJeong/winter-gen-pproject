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
Supports generative tasks by removing residues.
- Percentage: ~5% removed (args.crop_ratio=0.95).
- Detection: A random residue is picked; its nearest neighbors are kept.
- mask: [L] tensor (1.0 = visible, 0.0 = masked).
- torsion_mask: Reflects chemical validity; unaffected by spatial masking.

## 4. Item Dictionary
- F: Number of consecutive frames (defined by num_consecutive)
- L: Sequence length (number of residues)

Each item contains:
- name: Protein identifier (e.g., 4o66_C_R1).
- frame_indices: [num_consecutive] indices used.
- seqres: [L] sequence indices.
- mask: [L] spatial mask (0/1).
- torsion_mask: [L, 7] chemical validity.
- clean_trans: [F, L, 3] CA positions.
- clean_rots: [F, L, 3, 3] rotation matrices.
- clean_torsions: [F, L, 7, 2] torsion sin/cos.
- clean_atom37: [F, L, 37, 3] all atom coordinates.

## 5. Filtering
To avoid batching errors, always filter to a single protein:
- --pep_name: Filter by protein (e.g., "4o66_C"). Default: None. 
- --replica: Filter by replica (e.g., "1" for _R1). Default: 1.

## 6. Sampling
- num_consecutive: Frames per sample. Changes tensor shape (F dimension). Default: 1.
- stride: Gap between consecutive frames in a sample. Default: 1.
- repeat: Oversampling factor. Changes epoch size, not data shape. Default: 1.

## 7. Basic Usage
dataset = MDGenDataset(args, mode='train')
# Ensure args.pep_name is set for batching.
