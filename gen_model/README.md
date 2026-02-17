# Generative Model for Protein MD Trajectories

This directory contains code for training and running inference with diffusion models on molecular dynamics (MD) trajectories.

## Directory Structure

```
gen_model/
├── simple_train.py          # Basic DDPM training script
├── simple_inference.py      # Basic DDPM inference script
├── dataset.py               # MD trajectory data loader
├── models/                  # Neural network architectures
│   ├── score_network.py     # SE3 diffusion score network
│   ├── ipa_pytorch.py       # Invariant Point Attention
│   ├── nextnet.py           # NextNet backbone
│   └── layers.py            # Utility layers
├── diffusion/              # Diffusion implementations
│   ├── se3_diffuser.py     # SE3 diffusion (future use)
│   ├── so3_diffuser.py     # SO(3) rotation diffusion
│   ├── r3_diffuser.py      # R³ translation diffusion
│   └── igso3.py            # IGSO3 distribution
├── splits/                 # Train/val/test splits
└── utils/                  # Utility functions
```

## Quick Start

### Training

```bash
python gen_model/simple_train.py
```

### Inference

```bash
# Test on dataset
python gen_model/simple_inference.py \
  --checkpoint checkpoints/simple_ddpm/checkpoint_epoch_100.pt \
  --mode test \
  --num_samples 5

# Generate from noise
python gen_model/simple_inference.py \
  --checkpoint checkpoints/simple_ddpm/checkpoint_epoch_100.pt \
  --mode sample
```

---

## Dataset (MDGenDataset)

The `dataset.py` module provides `MDGenDataset` for loading MD trajectory data.

### Key Features

**SE(3) Invariance**: Every frame is aligned to frame 0 via global superposition using heavy atoms (non-hydrogen protein atoms) to ensure a consistent coordinate system across the trajectory.

**4-Way Split Logic**: Data is partitioned chronologically into:
- `train_early`: Frames [0 to main training end], includes early and main data
- `train`: Main training frames after the early threshold
- `val`: Validation frames
- `test`: Test frames

Default early threshold: 5ns. Default mode: 'train'.

**Spatial Masking**: Supports generative tasks by removing edge residues similar to image cropping. Only applied in 'train'/'train_early' when `crop_ratio` < 1.0.
- Goal: Uniform visibility probability (k/L) for all residues, removing core bias
- Mechanism: Iterative Proportional Fitting (IPF) computes balanced seed weights
- Reference Frame: Frame 0 (train_early) or First Train Frame (train)
- Selection: Weighted seed sampling → N nearest CA-CA neighbors kept
- Output: mask [L] tensor (1.0 = visible, 0.0 = masked)

### Dataset Output Format

Each item (after default collation) contains:
- **F**: Number of consecutive frames (defined by num_consecutive)
- **L**: Sequence length (number of residues)
- **B**: Batch size

```python
{
    'name': [1],                    # List of protein identifiers (deduplicated)
    'frame_indices': [B, F],        # Indices used
    'seqres': [1, L],               # Sequence indices (deduplicated)
    'mask': [B, L],                 # Spatial mask (0/1)
    'torsion_mask': [1, L, 7],      # Chemical validity (deduplicated)
    'clean_trans': [B, F, L, 3],    # CA positions
    'clean_rots': [B, F, L, 3, 3],  # Rotation matrices
    'clean_torsions': [B, F, L, 7, 2],  # Torsion sin/cos
    'clean_atom37': [B, F, L, 37, 3],   # All atom coordinates
}
```

### Filtering

To avoid batching errors, always filter to a single protein:
- `--pep_name`: Filter by protein (e.g., "4o66_C"). Default: None
- `--replica`: Filter by replica (e.g., "1" for _R1). Default: 1
- `--crop_ratio`: Fraction of residues to keep (0.0-1.0). Default: 0.95

### Sampling

- `num_consecutive`: Frames per sample. Changes tensor shape (F dimension). Default: 1
- `stride`: Gap between consecutive frames in a sample. Default: 1
- `repeat`: Oversampling factor. Changes epoch size, not data shape. Default: 1

### Basic Usage

```python
from gen_model.dataset import MDGenDataset
from omegaconf import OmegaConf

args = OmegaConf.create({
    'data_dir': 'data',
    'atlas_csv': 'data/atlas.csv',
    'train_split': 'gen_model/splits/frame_splits.csv',
    'suffix': '_latent',
    'frame_interval': None,
    'crop_ratio': 0.95,
    'min_t': 0.01,
})

dataset = MDGenDataset(
    args=args,
    diffuser=None,  # No SE3 diffuser needed for basic DDPM
    split=args.train_split,
    mode='train',
    repeat=1,
    num_consecutive=1,
    stride=1
)

# Ensure args.pep_name is set for batching
sample = dataset[0]
```

---

## DDPM Implementation

### SimpleDDPM

Basic DDPM implementation for frame denoising. Features:
- Linear noise schedule with β values
- Forward diffusion: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
- Reverse diffusion (denoising)
- ε-prediction (noise prediction objective)

### SimpleDenoiseModel

Simple U-Net-like model for denoising frames:
- MLP encoder-decoder with skip connections
- Time embedding via sinusoidal encoding
- Input: noisy frame + timestep
- Output: predicted noise

### Training Process

1. Load single frame from MD trajectory
2. Sample random timestep t ∈ [0, 1000]
3. Add Gaussian noise: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
4. Model predicts noise: ε_pred = model(x_t, t)
5. Loss: MSE(ε_pred, ε)
6. Save checkpoints every 10 epochs

### Inference Process

1. Start from noise (or noisy frame)
2. Reverse diffusion: x_{t-1} = denoise_step(x_t, t)
3. Iterate for all timesteps
4. Return denoised frame

---

## Hyperparameters

See [HYPERPARAMETER_GUIDE.md](../HYPERPARAMETER_GUIDE.md) for detailed tuning guide.

**Quick Reference:**
- `timesteps`: 100-1000 (diffusion steps)
- `hidden_dim`: 128-512 (model capacity)
- `batch_size`: 4-16 (GPU memory dependent)
- `num_epochs`: 50-200 (training duration)
- `learning_rate`: 1e-5 to 5e-4
- `beta_start`: 0.0001 (initial noise)
- `beta_end`: 0.02 (final noise)

---

## Google Colab

See [NOTEBOOK_USAGE.md](../NOTEBOOK_USAGE.md) for complete Colab workflow.

Quick start:
1. Push code to GitHub
2. Open `colab_single_protein_ddpm.ipynb` in Colab
3. Update `REPO_URL` in Step 2
4. Configure protein in Step 3
5. Run all cells

---

## Advanced: SE3 Diffusion (Future)

The `diffusion/` directory contains SE3 diffusion implementations for future use:
- `se3_diffuser.py`: Combined rotation + translation diffusion
- `so3_diffuser.py`: Rotation diffusion on SO(3)
- `r3_diffuser.py`: Translation diffusion on R³
- `igso3.py`: IGSO(3) distribution computations

The `models/` directory contains advanced architectures:
- `score_network.py`: SE3 score prediction network
- `ipa_pytorch.py`: Invariant Point Attention
