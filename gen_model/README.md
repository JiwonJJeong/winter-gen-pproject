# Gen Model - Simplified DDPM for Protein Frame Denoising

This directory contains a simplified implementation of DDPM (Denoising Diffusion Probabilistic Models) for training and inference on single protein frames from MD trajectories.

## Overview

The goal is to train a model that:
1. Takes a single protein frame
2. Adds noise to it following a diffusion schedule
3. Learns to denoise it back to the original frame

This is a basic DDPM implementation that serves as a foundation for more complex protein generation tasks.

## Directory Structure

```
gen_model/
├── train.py              -> symlink to simple_train.py (main training script)
├── inference.py          -> symlink to simple_inference.py (main inference script)
├── simple_train.py       # Standalone DDPM training implementation
├── simple_inference.py   # Standalone DDPM inference implementation
├── dataset.py            # MDGenDataset for loading MD trajectory data
├── models/               # Neural network models
│   ├── score_network.py  # SE3 score network (for advanced SE3 diffusion)
│   ├── ipa_pytorch.py    # Invariant Point Attention module
│   ├── layers.py         # Network layers
│   ├── nextnet.py        # NextNet backbone (ConvNext-based)
│   └── unet.py           # U-Net backbone
├── diffusion/            # Diffusion implementations
│   ├── diffusion.py      # Basic DDPM (used by main.py - image/video)
│   ├── conditional_diffusion.py  # Conditional DDPM
│   ├── se3_diffuser.py   # SE3 diffusion for protein structure
│   └── ...               # Other diffusion utilities
├── geometry.py           # Geometric transformations
├── residue_constants.py  # Protein residue definitions
├── rigid_utils.py        # Rigid body transformations
└── splits/               # Train/val/test split definitions
```

## Quick Start

### Training

Train a basic DDPM model on single frames:

```bash
python gen_model/train.py
```

This will:
- Load MD trajectory data from `data/` directory
- Train a denoising model to recover original frames from noised versions
- Save checkpoints to `checkpoints/simple_ddpm/`

### Inference

After training, run inference to test denoising:

```bash
# Test on dataset samples
python gen_model/inference.py --checkpoint checkpoints/simple_ddpm/checkpoint_epoch_100.pt --mode test --num_samples 5

# Generate new samples from noise
python gen_model/inference.py --checkpoint checkpoints/simple_ddpm/checkpoint_epoch_100.pt --mode sample
```

## Implementation Details

### SimpleDDPM Class

The `SimpleDDPM` class in `simple_train.py` implements:
- **Forward diffusion**: Gradually adds Gaussian noise to data
- **Noise schedule**: Linear schedule from β_start to β_end
- **Timesteps**: 1000 steps by default

### SimpleDenoiseModel

A simple MLP-based architecture that:
- Takes noisy frame and timestep as input
- Embeds the timestep
- Processes through encoder-decoder with skip connections
- Outputs predicted noise (epsilon-prediction)

### Training Process

1. Load frame from dataset
2. Sample random timestep t
3. Add noise according to diffusion schedule
4. Model predicts the noise
5. Compute MSE loss between predicted and actual noise
6. Update model parameters

### Inference Process

1. Start from random noise (or noisy frame)
2. Iteratively denoise for T timesteps
3. At each step, predict noise and compute less noisy version
4. Return final denoised frame

## Configuration

Edit the configuration in `simple_train.py`:

```python
args = OmegaConf.create({
    'data_dir': 'data',
    'atlas_csv': 'data/atlas.csv',
    'train_split': 'gen_model/splits/frame_splits.csv',
    'suffix': '_latent',
    'frame_interval': None,
    'crop_ratio': 0.95,
    'min_t': 0.01,
})
```

## Data Format

The MDGenDataset expects:
- MD trajectory data as `.npy` files in `data/` directory
- Atlas CSV mapping protein names to sequences
- Split CSV defining train/val/test splits

Each frame should contain atomic coordinates for all residues.

## Removed/Archived Files

For simplification, the following have been removed or archived:
- `experiments/` - Complex SE3 diffusion training (protein-specific, SE3 equivariant)
- `model/` - Duplicate of `gen_model/models/`
- `main.py` - Image/video DDPM (not needed for protein frames)
- `train_se3.py.bak` - Original SE3 training script (archived)
- `inference_se3.py.bak` - Original SE3 inference script (archived)

If you need SE3 equivariant diffusion for full protein structure generation, refer to the `.bak` files.

## Next Steps

This basic DDPM serves as a starting point. Potential enhancements:
1. **SE3 Equivariance**: Use `se3_diffuser.py` for rotation/translation equivariance
2. **Conditional Generation**: Condition on sequence, structure, or other features
3. **Improved Architecture**: Use attention mechanisms or graph neural networks
4. **Multi-frame Generation**: Extend to generate trajectories (multiple consecutive frames)

## References

- DDPM: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- SE(3) Diffusion: [SE(3) Diffusion for Protein Generation](https://arxiv.org/abs/2302.02277)
