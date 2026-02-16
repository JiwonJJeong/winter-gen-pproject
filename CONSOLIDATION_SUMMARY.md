# Code Consolidation Summary

## Goal
Consolidate repetitive code and create a basic DDPM training/inference setup for one frame to itself noised.

## Changes Made

### 1. Created Simplified DDPM Implementation

**New Files:**
- [gen_model/simple_train.py](gen_model/simple_train.py) - Basic DDPM training script
- [gen_model/simple_inference.py](gen_model/simple_inference.py) - Basic DDPM inference script  
- [gen_model/README.md](gen_model/README.md) - Documentation for the simplified setup

**Symlinks:**
- [gen_model/train.py](gen_model/train.py) → simple_train.py
- [gen_model/inference.py](gen_model/inference.py) → simple_inference.py

### 2. Removed Repetitive Code

**Deleted Directories:**
- `experiments/` - Complex SE3 protein diffusion training (SE3 equivariant, Hydra configs)
  - `train_se3_diffusion.py` (834 lines)
  - `inference_se3_diffusion.py` (492 lines)
  - `utils.py`
- `model/` - Duplicate of gen_model/models/
  - `score_network.py` (identical to gen_model/models/score_network.py with old imports)
  - `ipa_pytorch.py` (identical to gen_model/models/ipa_pytorch.py with old imports)
  - `layers.py` (identical to gen_model/models/layers.py)

**Deleted Files:**
- `main.py` - Image/video DDPM using PyTorch Lightning (not needed for protein frames)

**Archived Files:**
- [gen_model/train_se3.py.bak](gen_model/train_se3.py.bak) - Original SE3 training (for reference)
- [gen_model/inference_se3.py.bak](gen_model/inference_se3.py.bak) - Original SE3 inference (for reference)

### 3. Simplified Architecture

**Before:**
```
.
├── experiments/          # SE3 training with Hydra/Wandb
│   ├── train_se3_diffusion.py
│   └── inference_se3_diffusion.py
├── model/                # Duplicate models
│   ├── score_network.py
│   ├── ipa_pytorch.py
│   └── layers.py
├── gen_model/
│   ├── train.py          # SE3 training (200 lines)
│   ├── inference.py      # SE3 inference (492 lines)
│   ├── models/           # Actual models
│   └── diffusion/        # SE3 diffuser, etc.
└── main.py               # Image/video DDPM
```

**After:**
```
.
├── gen_model/
│   ├── train.py -> simple_train.py          # Basic DDPM (clean entry point)
│   ├── inference.py -> simple_inference.py  # Basic DDPM inference
│   ├── simple_train.py       # Standalone DDPM training (330 lines)
│   ├── simple_inference.py   # Standalone DDPM inference (280 lines)
│   ├── dataset.py            # MD trajectory data loader
│   ├── models/               # Neural network models
│   │   ├── score_network.py  # SE3 score network (advanced)
│   │   ├── nextnet.py        # Simple backbone (can be used)
│   │   └── ...
│   ├── diffusion/            # Diffusion implementations
│   │   ├── se3_diffuser.py   # For future SE3 work
│   │   └── ...
│   └── README.md             # Documentation
└── (removed: experiments/, model/, main.py)
```

## Key Features of New Implementation

### SimpleDDPM (simple_train.py)
- **Clean Implementation**: ~330 lines vs 834 lines in old SE3 version
- **No External Config**: No Hydra/Wandb dependencies required
- **Standard PyTorch**: Uses pure PyTorch (no PyTorch Lightning)
- **Frame Denoising**: Takes single frame, adds noise, learns to denoise
- **Basic Architecture**: Simple MLP encoder-decoder with skip connections

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

## Usage

### Training
```bash
python gen_model/train.py
```

### Inference
```bash
# Test on dataset
python gen_model/inference.py \
  --checkpoint checkpoints/simple_ddpm/checkpoint_epoch_100.pt \
  --mode test \
  --num_samples 5

# Generate from noise
python gen_model/inference.py \
  --checkpoint checkpoints/simple_ddpm/checkpoint_epoch_100.pt \
  --mode sample
```

## Benefits

1. **Reduced Complexity**: Removed 1500+ lines of complex SE3/Hydra code
2. **Clear Focus**: Single purpose - frame denoising with basic DDPM  
3. **Easy to Extend**: Simple codebase to build upon
4. **No Duplication**: Single source of truth for models in gen_model/models/
5. **Better Organization**: All protein-related code in gen_model/
6. **Preserved SE3 Code**: Backed up for future use (.bak files)

## Next Steps

To extend this basic DDPM:
1. **SE3 Equivariance**: Use se3_diffuser.py from gen_model/diffusion/
2. **Better Models**: Try nextnet.py or score_network.py from gen_model/models/
3. **Conditional Generation**: Add conditioning on sequence/structure
4. **Multi-frame**: Generate trajectories (consecutive frames)

## File Locations

- Training: [gen_model/train.py](gen_model/train.py) or [gen_model/simple_train.py](gen_model/simple_train.py)
- Inference: [gen_model/inference.py](gen_model/inference.py) or [gen_model/simple_inference.py](gen_model/simple_inference.py)
- Documentation: [gen_model/README.md](gen_model/README.md)
- Dataset: [gen_model/dataset.py](gen_model/dataset.py)
- Models: [gen_model/models/](gen_model/models/)
- Diffusion: [gen_model/diffusion/](gen_model/diffusion/)
