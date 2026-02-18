# Generative Model for Protein MD Trajectories

SE(3) score-matching diffusion on protein backbone rigid frames, trained on MD trajectory data.

---

## Directory Structure

```
gen_model/
├── simple_train.py          # SE(3) Lightning training module (SE3Module)
├── simple_inference.py      # Inference script
├── dataset.py               # MD trajectory data loader (MDGenDataset)
├── models/
│   ├── score_network.py     # ScoreNetwork: Embedder + IpaScore
│   └── ipa_pytorch.py       # Invariant Point Attention + TorsionAngles
├── diffusion/
│   ├── se3_diffuser.py      # SE3Diffuser: combines SO(3) + R³
│   ├── so3_diffuser.py      # SO(3) rotation diffusion via IGSO(3)
│   └── r3_diffuser.py       # R³ translation diffusion via VP-SDE
├── splits/                  # Train/val/test CSV splits
├── geometry.py              # atom37↔frames, torsion angle extraction
├── all_atom.py              # Backbone reconstruction from frames + psi
├── rigid_utils.py           # Rigid / Rotation / Quaternion math
└── residue_constants.py     # Amino acid geometry constants
```

---

## End-to-End Data Flow

### 1. Dataset → Batch

`MDGenDataset.__getitem__` produces one sample per MD frame:

```
.npy file  →  atom14 coords [L, 14, 3]
           →  atom37 coords [L, 37, 3]          (via atom14_to_atom37)
           →  backbone frames [L]  (Rigid)       (via atom14_to_frames)
           →  torsion_angles_sin_cos [L, 7, 2]  (via atom37_to_torsions)
```

The 7 torsion angles per residue are (in order): ω, φ, ψ, χ1, χ2, χ3, χ4.
Index 2 (`torsion_angles_sin_cos[..., 2, :]`) is **ψ** (the only one the model predicts).

**Centering and scaling** (applied to translations before output):
```
centroid   = mean(CA positions)
atom14_pos = (atom14_pos - centroid) * coord_scale
atom37_pos = (atom37_pos - centroid) * coord_scale
rigids_0[..., 4:] = (rigids_0[..., 4:] - centroid) * coord_scale
```
`coord_scale = 1 / std(CA coords)` computed once from training data.

### 2. Spatial Cropping (training only)

When `crop_ratio < 1.0`, a random seed residue is chosen and the nearest
`k = int(L * crop_ratio)` residues (by CA–CA distance) form the visible crop:

```
mask [L]:  1.0 = visible (inside crop)
           0.0 = masked  (outside crop, not diffused, not supervised)
```

Seed weights are computed via **Iterative Proportional Fitting (IPF)** on the
reference frame so that every residue has equal probability of being included
across different crops.

### 3. SE(3) Forward Diffusion

`SE3Diffuser.forward_marginal(rigids_0, t, diffuse_mask=mask)` is called per
sample with `t ~ Uniform(min_t, 1.0)` during training and `t = 1.0` at
validation.

```
Rotations  →  SO3Diffuser:  IGSO(3) Brownian motion with logarithmic sigma schedule
Translations → R3Diffuser:  VP-SDE  (variance-preserving SDE)
```

For masked residues (`mask = 0`), the diffuser leaves them at their clean
values and sets their scores to zero:

```
rot_t[mask=0]    = rot_0        (not noised)
trans_t[mask=0]  = trans_0      (not noised)
rot_score[mask=0]   = 0
trans_score[mask=0] = 0
```

**Batch keys produced by the dataset:**

| Key | Shape | Description |
|-----|-------|-------------|
| `rigids_0` | `[L, 7]` | Clean frame (quat[4] + trans[3]) |
| `rigids_t` | `[L, 7]` | Noised frame at time t |
| `rot_score` | `[L, 3]` | SO(3) score target |
| `trans_score` | `[L, 3]` | R³ score target |
| `rot_score_scaling` | scalar | Score normalisation for this t |
| `trans_score_scaling` | scalar | Score normalisation for this t |
| `t` | scalar | Noise level in [0, 1] |
| `res_mask` | `[L]` | Spatial crop mask (1=visible) |
| `fixed_mask` | `[L]` | Motif mask (all 0 here — everything is diffused) |
| `torsion_angles_sin_cos` | `[L, 7, 2]` | Ground-truth torsions |
| `seq_idx` | `[L]` | 1-indexed residue positions |
| `sc_ca_t` | `[L, 3]` | Self-conditioning CA positions (zeros initially) |
| `atom14_pos` | `[L, 14, 3]` | Clean atom positions (centred + scaled) |
| `atom37_pos` | `[L, 37, 3]` | Clean all-atom positions (centred + scaled) |
| `aatype` | `[L]` | Amino acid type indices |

### 4. ScoreNetwork Forward Pass

`ScoreNetwork.forward(batch)`:

```
seq_idx, t, fixed_mask, sc_ca_t
        │
        ▼
    Embedder
        │  node_embed [B, N, node_embed_size]
        │  edge_embed [B, N, N, edge_embed_size]
        ▼
    IpaScore  (num_blocks × IPA + Transformer + BackboneUpdate)
        │  Uses rigids_t as initial frames
        │  Updates frames via rigid body composition (masked by diffuse_mask)
        │
        ├──→ calc_rot_score(init_rots, curr_rots, t)   → rot_score  [B, N, 3]
        ├──→ calc_trans_score(init_trans, curr_trans, t) → trans_score [B, N, 3]
        ├──→ TorsionAngles(node_embed)                  → psi [B, N, 2]  (sin/cos)
        └──→ BackboneUpdate → final_rigids
        │
        ▼
    ScoreNetwork applies:
        psi_pred = model_psi  (since fixed_mask=0, always predicted)
        atom37, atom14 = compute_backbone(final_rigids, psi_pred)
```

**Output dict:** `rot_score`, `trans_score`, `psi`, `rigids`, `atom37`, `atom14`

The `rot_score` and `trans_score` outputs are zeroed for masked residues
(`* node_mask`) inside `IpaScore`.

### 5. What the Model Actually Reads from the Batch

Despite the dataset producing ~20 keys, `ScoreNetwork.forward` consumes exactly **7**:

| Key | Shape | Used for |
|-----|-------|---------|
| `rigids_t` | `[B, N, 7]` | Starting SE(3) frames that IpaScore iteratively refines |
| `res_mask` | `[B, N]` | Gates node/edge embeddings; zeroes masked residue outputs |
| `fixed_mask` | `[B, N]` | Controls which residues get frame updates (all zeros → all free) |
| `seq_idx` | `[B, N]` | Positional index embedding in Embedder |
| `t` | `[B]` | Timestep sinusoidal embedding in Embedder; also used inside `calc_rot_score` / `calc_trans_score` |
| `sc_ca_t` | `[B, N, 3]` | Self-conditioning CA positions fed to Embedder (zeros on first pass) |
| `torsion_angles_sin_cos` | `[B, N, 7, 2]` | `[..., 2, :]` (ψ only) — blended with predicted ψ for fixed residues (no-op here since `fixed_mask=0`) |

The loss additionally reads `rot_score`, `trans_score`, `rot_score_scaling`, `trans_score_scaling`, and `res_mask`.

Keys produced by the dataset but **not consumed during training**:
`rigids_0`, `atom14_pos`, `atom37_pos`, `aatype`, `chain_idx`, `residue_index`,
`residx_atom14_to_atom37`, `atom37_mask`, `centroid`, `coord_scale`, `name`, `frame_indices`
— retained for inference and evaluation (RMSD, coordinate unscaling, etc.).

---

### 6. Loss (SE3Module)

All three terms are computed only over visible residues (`res_mask = 1`):

```python
n_visible = mask.sum()

# Rotation: scale-normalised MSE on SO(3) score vectors
rot_loss = Σ_visible  ||pred_rot_score / rot_scaling - gt_rot_score / rot_scaling||²  / n_visible

# Translation: scale-normalised MSE on R³ score vectors
trans_loss = Σ_visible  ||pred_trans_score / trans_scaling - gt_trans_score / trans_scaling||²  / n_visible

# Psi torsion: sin/cos MSE (unit-normalised by TorsionAngles layer)
psi_loss = Σ_visible  ||pred_psi - gt_psi||²  / n_visible   # gt_psi = torsion_angles_sin_cos[..., 2, :]

total_loss = rot_weight * rot_loss + trans_weight * trans_loss + psi_weight * psi_loss
```

Scale normalisation (`rot_score_scaling`, `trans_score_scaling`) equalises the
loss magnitude across different noise levels t.

**Logged metrics:** `train_loss`, `train_rot_loss`, `train_trans_loss`,
`train_psi_loss`, and their `val_*` equivalents.

---

## Quick Start

### Training (CLI)

```bash
python gen_model/simple_train.py \
  --data_dir data \
  --atlas_csv gen_model/splits/atlas.csv \
  --train_split gen_model/splits/frame_splits.csv \
  --suffix _latent \
  --batch_size 8 \
  --epochs 100 \
  --lr 1e-4 \
  --save_dir checkpoints/se3
```

### Training (notebook)

Open `colab_single_protein_ddpm.ipynb`. The key config knobs are in cell 7
(`protein_config`). Run all cells sequentially.

### Usage in code

```python
from omegaconf import OmegaConf
from gen_model.dataset import MDGenDataset
from gen_model.diffusion.se3_diffuser import SE3Diffuser
from gen_model.simple_train import SE3Module

se3_conf = OmegaConf.create({...})    # see simple_train.py main() for defaults
model_conf = OmegaConf.create({...})  # see simple_train.py main() for defaults

diffuser = SE3Diffuser(se3_conf)

train_dataset = MDGenDataset(args=data_args, diffuser=diffuser, mode='train')
val_dataset   = MDGenDataset(args=data_args, diffuser=diffuser, mode='val')
val_dataset.coord_scale = float(train_dataset.coord_scale)

model = SE3Module(model_conf=model_conf, se3_conf=se3_conf, lr=1e-4)
```

---

## Key Configuration

### SE3Diffuser (`se3_conf`)

| Key | Default | Description |
|-----|---------|-------------|
| `diffuse_rot` | `True` | Enable SO(3) rotation diffusion |
| `diffuse_trans` | `True` | Enable R³ translation diffusion |
| `so3.schedule` | `logarithmic` | Sigma schedule for IGSO(3) |
| `so3.min_sigma` | `0.1` | Minimum rotation noise level |
| `so3.max_sigma` | `1.5` | Maximum rotation noise level |
| `r3.min_b` | `0.1` | VP-SDE β start |
| `r3.max_b` | `20.0` | VP-SDE β end |
| `r3.coordinate_scaling` | `0.1` | Ångström → normalised units |

### ScoreNetwork (`model_conf`)

| Key | Default | Description |
|-----|---------|-------------|
| `node_embed_size` | `256` | Single representation dim (= `ipa.c_s`) |
| `edge_embed_size` | `128` | Pair representation dim (= `ipa.c_z`) |
| `ipa.num_blocks` | `4` | Number of IPA + Transformer blocks |
| `ipa.no_heads` | `12` | IPA attention heads |
| `ipa.seq_tfmr_num_layers` | `2` | Transformer layers per block |

### Loss weights (`training`)

| Key | Default | Controls |
|-----|---------|---------|
| `rot_loss_weight` | `1.0` | SO(3) score matching |
| `trans_loss_weight` | `1.0` | R³ score matching |
| `psi_loss_weight` | `1.0` | ψ torsion angle reconstruction |
