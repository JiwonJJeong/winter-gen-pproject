# SE(3) Diffusion Integration for MD Trajectory Data

This directory contains the integrated dataset and training code for SE(3) diffusion models on MD trajectory data.

## Files

- **`dataset.py`**: Modified `MDGenDataset` with SE(3) diffusion integration
- **`test_dataset.py`**: Test script to verify dataset outputs
- **`train.py`**: Training script template
- **`geometry.py`**, **`rigid_utils.py`**, **`residue_constants.py`**: Geometry utilities

## Quick Start

### 1. Test the Dataset

First, verify that your dataset outputs all required features:

```bash
python gen_model/test_dataset.py
```

Update the paths in `test_dataset.py`:
- `data_dir`: Your MD trajectory data directory
- `atlas_csv`: Path to your atlas.csv file
- `split`: Path to your frame splits CSV

### 2. Train the Model

```bash
python gen_model/train.py
```

Update the configuration in `train.py` as needed.

## Key Changes from Original Dataset

### Added Features

The dataset now outputs all features required by the SE(3) diffusion model:

**Core Identifiers:**
- `aatype`: Amino acid types [L]
- `seq_idx`: Residue indices [L]
- `chain_idx`: Chain identifiers [L]
- `res_mask`: Residue mask [L]
- `fixed_mask`: Fixed residues (motifs) [L]

**Ground Truth (t=0):**
- `rigids_0`: Clean rigid transforms [L, 7]
- `atom37_pos`: All-atom positions [L, 37, 3]
- `atom14_pos`: Atom14 positions [L, 14, 3]
- `torsion_angles_sin_cos`: Torsion angles [L, 7, 2]

**Noised Structure (at time t):**
- `rigids_t`: Noised rigid transforms [L, 7]
- `rot_score`: Rotation score (ground truth) [L, 3]
- `trans_score`: Translation score (ground truth) [L, 3]
- `rot_score_scaling`: Rotation score scaling factor
- `trans_score_scaling`: Translation score scaling factor
- `t`: Time step in [0, 1]

**Metadata:**
- `sc_ca_t`: Self-conditioning CA positions [L, 3]
- `residx_atom14_to_atom37`: Atom mapping [L, 14]
- `atom37_mask`: Atom37 mask [L, 37]
- `residue_index`: Original residue indices [L]

### Usage Example

```python
from omegaconf import OmegaConf
from gen_model.dataset import MDGenDataset
from data_se3.data.se3_diffuser import SE3Diffuser

# Create SE3Diffuser
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
        'coordinate_scaling': 0.1,
    }
})
diffuser = SE3Diffuser(se3_conf)

# Create dataset
args = OmegaConf.create({
    'data_dir': 'data',
    'atlas_csv': 'data/atlas.csv',
    'suffix': '_latent',
    'frame_interval': None,
    'crop_ratio': 0.95,
    'min_t': 0.01,
})

dataset = MDGenDataset(
    args=args,
    diffuser=diffuser,
    split='gen_model/splits/frame_splits.csv',
    mode='train',
)

# Get a sample
sample = dataset[0]
```

## Dependencies

- PyTorch
- NumPy
- Pandas
- OmegaConf
- OpenFold (optional, for data_transforms)

## Notes

- The dataset uses the **first frame** from multi-frame trajectories for SE(3) diffusion
- Spatial cropping is applied during training (controlled by `crop_ratio`)
- Diffusion noise is applied via `SE3Diffuser.forward_marginal()`
- Self-conditioning is initially disabled but can be enabled in the model config
