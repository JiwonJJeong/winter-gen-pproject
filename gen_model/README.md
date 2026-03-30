# STAR-MD: Single-Trajectory SE(3) Diffusion for Protein MD

SE(3) score-matching diffusion on protein backbone rigid frames, combining
the **STAR-MD** network architecture with **SinFusion**'s single-trajectory
training protocol. Trained on one MD trajectory, generates new conformations
(unconditional) or entire trajectories (conditional, autoregressive).

---

## Method Overview

### Our Approach

We adapt two published methods for single-trajectory protein MD generation:

- **STAR-MD** (Shoghi et al., ICLR 2026) — spatio-temporal attention with
  block-causal masking, 2D-RoPE, AdaLN conditioning, and Diffusion Forcing
  for multi-frame training.
- **SinFusion** (Nikankin et al., ICML 2023) — single-video DDPM training
  protocol with virtual epochs, batch size 1, curriculum learning, and
  spatial augmentation to prevent overfitting.

The model trains on a single replica of a single protein's MD trajectory
and learns to generate new conformations that match the equilibrium
distribution and dynamics of that protein.

### Comparison: Our Model vs. MDGen vs. MDGen Naive

| Aspect | **Our Model** | **MDGen** (pre-trained) | **MDGen Naive** (single-traj) |
|--------|---------------|-------------------------|-------------------------------|
| **Diffusion** | SE(3) score matching (VP-SDE on SO(3)×R³) | Flow matching (linear coupling, velocity prediction) | Flow matching (same as MDGen) |
| **Architecture** | IPA → **joint** spatio-temporal attention (N×L tokens, block-causal, 2D-RoPE) | IPA → **separate** spatial attn → temporal attn (sequential) | Same as MDGen |
| **Representation** | SE(3) rigid frames (7-dim: quaternion + translation) | Latent 21-dim (7 frame offsets + 14 torsion sin/cos) | Same as MDGen |
| **Training data** | 1 trajectory, 1 protein | 1266 ATLAS proteins | 1 trajectory, 1 protein |
| **Training protocol** | SinFusion: virtual epoch=5000, batch=1, stratified t, curriculum for δt, context noise | Standard: large dataset, batch=1, uniform t | Standard: same as MDGen (no anti-overfitting) |
| **Conditioning** | AdaLN on [t_emb ∥ δt_emb], sc_ca_t (prev frame CA) | Latent conditioning (frame 0 offsets) | Same as MDGen |
| **Inference** | Reverse SDE, autoregressive with context noise (Diffusion Forcing) | ODE solver (Dopri5), autoregressive | Same as MDGen |
| **Positional encoding** | 2D-RoPE (residue × frame) | Separate RoPE per axis | Same as MDGen |

### Why "MDGen Naive" as a baseline?

Training MDGen's own architecture on one trajectory without any
anti-overfitting techniques isolates the contribution of:
1. **STAR-MD architecture** — joint ST attention vs. separate spatial/temporal
2. **SinFusion protocol** — virtual epochs, curriculum, batch=1, stratified noise
3. **SE(3) score matching** — vs. flow matching

If MDGen Naive overfits badly while our model doesn't, the SinFusion
anti-overfitting techniques are the key. If both perform similarly, the
architectural differences (joint vs. separate attention) matter more.

---

## Anti-Overfitting Techniques (SinFusion Protocol)

| Technique | What it does | Default |
|-----------|-------------|---------|
| **Virtual epoch size** | Reports 5000 samples/epoch regardless of trajectory length; each sample is a different random crop + noise level | `--virtual_epoch_size 5000` |
| **Batch size 1** | Maximum stochasticity per update; works with LayerNorm (not BatchNorm) | `--batch_size 1` |
| **Stratified t sampling** | Divides [min_t, max_t] into B strata, one sample per stratum — uniform noise coverage | In `SE3Diffusion.forward()` |
| **Curriculum for δt** | Phase 1 (<75k): δt ∈ [0.01, 0.1] ns; Phase 2 (75k-150k): [0.01, 1.0]; Phase 3: [0.01, 10.0] | `--curriculum` |
| **Spatial cropping** | Random 95% spatial crop with IPF-balanced seed weights | `crop_ratio=0.95` |
| **Context noise** | All L frames noised at τ ∼ U[0.01, 0.1] during training; at inference τ ∼ U[0, 0.05] on history frames | Diffusion Forcing |
| **EMA** | Exponential moving average of weights (decay 0.999) for validation/inference | `--ema_decay 0.999` |
| **Gradient clipping** | Norm clipping at 1.0 | `--grad_clip 1.0` |

---

## Directory Structure

```
gen_model/
├── train_unconditional.py       # Unconditional training (SinFusion protocol)
├── train_conditional.py         # Conditional STAR-MD training (Diffusion Forcing)
├── train_base.py                # Shared config helpers
├── se3_diffusion_module.py      # Lightning modules (SE3Diffusion, ConditionalSE3Diffusion, EMA)
├── inference_unconditional.py   # Generate conformations from noise
├── inference_conditional.py     # Autoregressive trajectory rollout
├── evaluate.py                  # Evaluation suite (torsion JSD, TICA JSD, autocorrelation)
├── hpo.py                       # Optuna hyperparameter search
├── dataset.py                   # Backward-compat shim
├── data/
│   ├── dataset.py               # MDGenDataset, ConditionalMDGenDataset
│   ├── geometry.py              # atom14↔atom37↔frames, torsion extraction
│   ├── all_atom.py              # Backbone reconstruction from frames + psi
│   └── residue_constants.py     # Amino acid geometry constants (extended)
├── models/
│   ├── star_score_network.py    # StarScoreNetwork: Embedder + StarIpaScore
│   ├── star_ipa.py              # StarIpaScore: IPA + joint ST attention
│   ├── star_attention.py        # SpatioTemporalAttention (block-causal, 2D-RoPE, KV-cache)
│   ├── rope2d.py                # 2D Rotary Position Embedding
│   ├── adaln.py                 # Adaptive Layer Normalization
│   └── lora.py                  # LoRA adapter
├── diffusion/
│   ├── se3_diffuser.py          # Shim → extern/se3_diffusion
│   ├── r3_diffuser.py           # Shim → extern/se3_diffusion
│   └── so3_diffuser.py          # Shim → extern/se3_diffusion
├── utils/
│   ├── rigid_utils.py           # Rigid / Rotation / Quaternion math
│   ├── tensor_utils.py          # Batched gather
│   └── structure_utils.py       # PDB I/O, CA trajectory conversion
└── splits/                      # atlas.csv, frame_splits.csv
```

---

## Quick Start

### Training

```bash
# Unconditional (generate conformations from noise)
python gen_model/train_unconditional.py \
    --protein 4o66_C --replica 1 --data_dir data

# Conditional (autoregressive trajectory generation)
python gen_model/train_conditional.py \
    --protein 4o66_C --replica 1 --data_dir data
```

### Inference

```bash
# Unconditional: 100 i.i.d. samples
python gen_model/inference_unconditional.py \
    --checkpoint checkpoints/unconditional/4o66_C/last.ckpt \
    --npy_path data/4o66_C/4o66_C_R1_latent.npy \
    --protein_name 4o66_C --num_samples 100

# Conditional: 250-frame trajectory (ATLAS protocol)
# Default: --n_steps 150. Initial run used --n_steps 200.
python gen_model/inference_conditional.py \
    --checkpoint checkpoints/conditional/4o66_C/last.ckpt \
    --data_dir data --protein 4o66_C --total_frames 250 --n_steps 200
```

### Evaluation

```bash
# Evaluate against all 3 replicas
python gen_model/evaluate.py \
    --ref_npy data/4o66_C/4o66_C_R{1,2,3}_latent.npy \
    --gen_traj outputs/conditional/4o66_C/traj.pt \
    --protein 4o66_C --mode conditional
```

### Colab Notebooks

- `colab_train_unconditional.ipynb` — full pipeline: train → generate → evaluate → generalization test → MDGen baselines
- `colab_train_conditional.ipynb` — same, with STAR-MD conditional training

---

## Evaluation Metrics

Following MDGen's analysis pipeline (`extern/mdgen/scripts/analyze_peptide_sim.py`):

| Metric | What it measures | Applies to |
|--------|-----------------|------------|
| **Torsion JSD** | Jensen-Shannon divergence on backbone (φ/ψ/ω) and sidechain (χ1-4) dihedral distributions | Both |
| **2D φ-ψ JSD** | Joint phi-psi distribution divergence | Both |
| **TICA JSD** | Divergence in time-lagged independent component space (slow collective motions) | Both |
| **Autocorrelation** | Temporal decorrelation of torsion angles and TICA components | Conditional only |

Evaluation defaults follow the ATLAS protocol: 250-frame trajectories
(conditional) or 100 i.i.d. samples (unconditional), compared against
all 3 reference MD replicas.

---

## Extern Submodules

| Submodule | Purpose |
|-----------|---------|
| `extern/se3_diffusion` | SE(3) diffuser, IPA, Embedder, backbone reconstruction |
| `extern/mdgen` | MDGen model/training/analysis (baseline + evaluation metrics) |
| `extern/sinfusion` | SinFusion reference (training protocol design) |

Initialize with: `git submodule update --init --recursive`
