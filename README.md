# Protein MD Trajectory Diffusion Model

This repository implements **DDPM (Denoising Diffusion Probabilistic Models)** for protein molecular dynamics (MD) trajectories. Train models to denoise and generate protein conformations from MD simulations.

---

## 🚀 Quick Start

### Local Training

```bash
# Train DDPM on single protein
python gen_model/simple_train.py
```

### Google Colab Training (Recommended)

1. **Push code to GitHub:**
   ```bash
   git add gen_model/ colab_single_protein_ddpm.ipynb
   git commit -m "Add DDPM training code"
   git push origin main
   ```

2. **Open in Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com/)
   - File → Open notebook → GitHub → Select `colab_single_protein_ddpm.ipynb`

3. **Run training:**
   - Enable GPU (Runtime → Change runtime type → GPU)
   - Update `REPO_URL` in Step 2
   - Configure protein in Step 3
   - Run all cells

📖 **Full Guide:** [NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md)

---

## 📂 Repository Structure

```
winter-gen-pproject/
├── gen_model/                          # Main code directory
│   ├── simple_train.py                # DDPM training script
│   ├── simple_inference.py            # DDPM inference script
│   ├── dataset.py                     # MD trajectory data loader
│   ├── models/                        # Neural network architectures
│   ├── diffusion/                     # Diffusion implementations
│   └── README.md                      # Code documentation
│
├── colab_single_protein_ddpm.ipynb    # Google Colab notebook
│
├── NOTEBOOK_USAGE.md                  # Complete Colab guide
├── HYPERPARAMETER_GUIDE.md            # Hyperparameter tuning guide
└── README.md                          # This file
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **[NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md)** | Complete guide for Google Colab training |
| **[HYPERPARAMETER_GUIDE.md](HYPERPARAMETER_GUIDE.md)** | Detailed hyperparameter tuning guide |
| **[gen_model/README.md](gen_model/README.md)** | Code documentation and API reference |

---

## 🎯 What This Does

- **Input:** Protein MD trajectory (atom positions over time)
- **Process:** Train a diffusion model to denoise protein conformations
- **Output:** Generated protein structures via reverse diffusion

### Key Features

- **Single-protein focus**: Train on specific protein trajectories
- **Google Colab ready**: Full workflow with GPU support
- **Git-based workflow**: Version control and collaboration built-in
- **Dynamic data creation**: No need to upload large data files
- **Configurable**: Easy hyperparameter tuning

---

## 🔧 Configuration

Configure training in `colab_single_protein_ddpm.ipynb` Step 3:

```python
protein_config = {
    'protein': {
        'name': '4o66_C',        # Your protein name
        'num_frames': 200,       # Trajectory frames
        'num_residues': 100,     # Protein size
    },
    'diffusion': {
        'timesteps': 500,        # Diffusion steps (100-1000)
    },
    'model': {
        'hidden_dim': 256,       # Model capacity (128-512)
    },
    'training': {
        'batch_size': 8,         # Batch size (4-16)
        'num_epochs': 100,       # Training epochs (50-200)
        'learning_rate': 1e-4,   # Learning rate (1e-5 to 5e-4)
    }
}
```

**Recommended Configurations:**

| Config | Timesteps | Hidden Dim | Epochs | Time (T4 GPU) |
|--------|-----------|------------|--------|---------------|
| Quick Test | 100 | 128 | 50 | 30-60 min |
| Balanced ⭐ | 500 | 256 | 100 | 2-4 hours |
| High Quality | 1000 | 512 | 200 | 8-12 hours |

See [HYPERPARAMETER_GUIDE.md](HYPERPARAMETER_GUIDE.md) for details.

---

## 📊 Dataset

The `MDGenDataset` loader provides:

- **SE(3) invariance**: Aligned frames for consistent coordinates
- **Spatial masking**: Edge residue cropping for generative tasks
- **4-way splits**: train_early, train, val, test
- **Flexible sampling**: Single or consecutive frames

**Basic Usage:**

```python
from gen_model.dataset import MDGenDataset
from omegaconf import OmegaConf

args = OmegaConf.create({
    'data_dir': 'data',
    'atlas_csv': 'data/atlas.csv',
    'train_split': 'gen_model/splits/frame_splits.csv',
    'crop_ratio': 0.95,
})

dataset = MDGenDataset(args=args, diffuser=None, mode='train')
```

See [gen_model/README.md](gen_model/README.md#dataset-mdgendataset) for details.

---

## 🧠 Model Architecture

### SimpleDDPM

Basic DDPM implementation:
- Linear noise schedule: β ∈ [0.0001, 0.02]
- Forward diffusion: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
- ε-prediction objective: predict noise from noisy input

### SimpleDenoiseModel

U-Net-like architecture:
- MLP encoder-decoder with skip connections
- Time embedding via sinusoidal encoding
- 3 encoder blocks + bottleneck + 3 decoder blocks

**Training:** MSE loss between predicted and actual noise
**Inference:** Iterative denoising from pure noise to clean structure

---

## 🎓 Examples

### Train Locally

```bash
python gen_model/simple_train.py
```

Checkpoints saved to `checkpoints/simple_ddpm/`

### Run Inference

```bash
# Test on validation set
python gen_model/simple_inference.py \
  --checkpoint checkpoints/simple_ddpm/checkpoint_epoch_100.pt \
  --mode test \
  --num_samples 5

# Generate from noise
python gen_model/simple_inference.py \
  --checkpoint checkpoints/simple_ddpm/checkpoint_epoch_100.pt \
  --mode sample
```

### Train on Google Colab

See [NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md) for complete workflow.

---

## 🔬 Advanced: SE3 Diffusion (Future)

The repository includes SE(3) equivariant diffusion implementations for future work:

- `gen_model/diffusion/se3_diffuser.py` - Combined rotation + translation diffusion
- `gen_model/models/score_network.py` - SE3 score prediction network
- `gen_model/models/ipa_pytorch.py` - Invariant Point Attention

Currently using basic Euclidean DDPM for simplicity. SE3 diffusion can be enabled by passing an SE3Diffuser to the dataset.

---

## 📝 Citation

If you use this code, please cite the relevant diffusion model papers:

- **DDPM**: [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- **SE3 Diffusion**: [SE(3) Diffusion for Protein Generation (Yim et al., 2023)](https://arxiv.org/abs/2302.02277)

---

## 📄 License

[Add your license here]

---

## 🤝 Contributing

Contributions welcome! Please open an issue or pull request.

---

## 🐛 Issues

If you encounter any issues:

1. Check [NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md#troubleshooting) troubleshooting section
2. Verify GPU is enabled in Colab
3. Check file paths and configuration
4. Open a GitHub issue with error details
