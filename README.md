# STAR-MD: Spatio-Temporal Attention for Protein MD Trajectory Generation

Generates realistic protein MD trajectories by combining **SE(3) score-matching diffusion** with joint spatio-temporal attention across frames, trained on a single MD trajectory following the **SinFusion** single-trajectory protocol.

---

## Prior works

| Work | What it does |
|------|-------------|
| [SE(3)-Diffusion (Yim et al., 2023)](https://arxiv.org/abs/2302.02277) | SE(3) score-matching diffusion on protein backbones; single-structure generation |
| [MDGen (Jing et al., 2024)](https://arxiv.org/abs/2407.01791) | Extends SE(3)-Diffusion to multi-frame MD trajectories via conditional generation |
| [SinFusion (Nikankin et al., 2022)](https://arxiv.org/abs/2211.11743) | Single-video/image diffusion with virtual epochs, curriculum, and spatial augmentation |

This repo combines all three: the STAR-MD architecture (joint ST attention, 2D-RoPE, AdaLN, block-causal masking) trained with SinFusion's single-trajectory protocol on SE(3) rigid frames.

---

## Setup

```bash
git clone --recurse-submodules <this-repo>
conda env create -f installation.yaml
conda activate star-md
```

The `extern/` submodules are read-only. Do not commit into them.

---

## Repo structure

```
winter-gen-pproject/
├── extern/                          # Read-only upstream submodules
│   ├── se3_diffusion/               # Yim et al. 2023
│   ├── mdgen/                       # Jing et al. 2024
│   └── sinfusion/                   # Nikankin et al. 2022
│
├── gen_model/
│   ├── path_setup.py                # Adds extern/ to sys.path; patches upstream dtype bugs
│   │
│   ├── models/
│   │   ├── star_score_network.py    # StarScoreNetwork: Embedder + StarIpaScore
│   │   ├── star_ipa.py              # StarIpaScore: non-invasive IpaScore extension
│   │   ├── star_attention.py        # SpatioTemporalAttention (block-causal, 2D-RoPE, KV-cache)
│   │   ├── rope2d.py                # 2D Rotary Position Embedding (residue × frame)
│   │   ├── adaln.py                 # Adaptive Layer Normalization conditioned on (t, Δt)
│   │   ├── score_network.py         # Shim → extern/se3_diffusion ScoreNetwork
│   │   └── lora.py                  # LoRA adapter injection
│   │
│   ├── diffusion/
│   │   ├── se3_diffuser.py          # Shim → extern/se3_diffusion SE3Diffuser
│   │   ├── so3_diffuser.py          # Shim → extern/se3_diffusion SO3Diffuser
│   │   └── r3_diffuser.py           # Shim → extern/se3_diffusion R3Diffuser
│   │
│   ├── data/
│   │   ├── dataset.py               # MDGenDataset + ConditionalMDGenDataset
│   │   ├── geometry.py              # atom14↔atom37↔frames, torsion extraction
│   │   ├── all_atom.py              # Backbone reconstruction from frames + psi
│   │   └── residue_constants.py     # Amino acid geometry constants
│   │
│   ├── se3_diffusion_module.py      # Lightning modules (SE3Diffusion, ConditionalSE3Diffusion)
│   ├── train_base.py                # Shared config helpers (default_se3_conf, default_model_conf)
│   ├── train_unconditional.py       # Stage 1 training
│   ├── train_conditional.py         # Stage 2 training with δt curriculum
│   ├── inference_unconditional.py   # Sample conformations from noise
│   ├── inference_conditional.py     # Autoregressive trajectory rollout
│   ├── evaluate.py                  # Evaluation suite (torsion JSD, TICA, autocorrelation)
│   │
│   └── splits/
│       ├── atlas.csv                # Protein name → sequence
│       └── frame_splits.csv         # Train/val/test frame boundaries
│
├── checkpoints/
│   ├── unconditional/               # Stage 1 checkpoints
│   └── conditional/                 # Stage 2 checkpoints
│
└── installation.yaml                # Conda environment
```

---

## Training

### Stage 1 — unconditional (single-frame score matching)

```bash
python gen_model/train_unconditional.py \
    --protein 4o66_C \
    --replica 1 \
    --data_dir data \
    --max_steps 200000 \
    --batch_size 1 \
    --lr 1e-4 \
    --ema_decay 0.999
```

### Stage 2 — conditional (multi-frame, Diffusion Forcing)

```bash
python gen_model/train_conditional.py \
    --protein 4o66_C \
    --replica 1 \
    --data_dir data \
    --max_steps 200000 \
    --num_frames 16 \
    --ns_per_stored_frame 0.1 \
    --curriculum
```

Key Stage 2 flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--num_frames` | 8 | Window size L (frames per training sample) |
| `--ns_per_stored_frame` | 0.1 | Physical time per stored MD frame (ns) |
| `--curriculum` / `--no_curriculum` | on | δt curriculum: [0.01,0.1] → [0.01,1.0] → [0.01,10.0] ns |
| `--star_enabled` / `--no_star` | on | Enable spatio-temporal attention blocks |
| `--st_num_heads` | 8 | Number of attention heads in ST blocks |
| `--spatial_sigma` | 0.0 | Gaussian spatial bias in ST attention (0 = disabled) |
| `--lora_r` | 0 | LoRA rank (0 = no LoRA, full fine-tune) |
| `--lora_alpha` | 0.0 | LoRA alpha scaling |
| `--grad_clip` | 1.0 | Gradient norm clip |
| `--ema_decay` | 0.999 | EMA weight decay (0 = disabled) |
| `--warmup_steps` | 0 | Linear LR warmup steps |
| `--virtual_epoch_size` | 5000 | Samples per virtual epoch (SinFusion anti-overfit) |

---

## Inference

### Conditional — autoregressive trajectory rollout

```bash
python gen_model/inference_conditional.py \
    --checkpoint checkpoints/conditional/last.ckpt \
    --data_dir data \
    --protein 4o66_C \
    --output generated_traj.pt \
    --total_frames 250 \
    --delta_t 0.1 \
    --n_steps 150
```

Output: `[T, N, 7]` tensor of backbone rigid frames (quaternion + translation, coordinate-scaled).

### Unconditional — sample conformations from noise

```bash
python gen_model/inference_unconditional.py \
    --checkpoint checkpoints/unconditional/last.ckpt \
    --npy_path data/4o66_C/4o66_C_R1_latent.npy \
    --protein_name 4o66_C \
    --num_samples 100 \
    --out_dir outputs/unconditional
```

Output: `[num_samples, N, 3]` CA coordinates in Ångströms.

---

## Evaluation

```bash
python gen_model/evaluate.py \
    --ref_npy "data/4o66_C/4o66_C_R1_latent.npy data/4o66_C/4o66_C_R2_latent.npy" \
    --gen_traj outputs/generated_traj.pt \
    --protein 4o66_C \
    --mode conditional
```

Metrics follow the MDGen/ATLAS protocol:

| Metric | What it measures |
|--------|-----------------|
| Torsion JSD | φ/ψ/ω and χ1-4 dihedral distributions |
| 2D φ-ψ JSD | Joint Ramachandran divergence |
| TICA JSD | Slow collective motion distribution |
| Autocorrelation | Temporal decorrelation of torsion/TICA (conditional only) |

---

## Architecture

**SpatioTemporalAttention** (`gen_model/models/star_attention.py`) injects one cross-frame attention block after each IPA block:
- **Block-causal mask**: frame ℓ attends only to frames ≤ ℓ, enabling autoregressive rollout
- **2D-RoPE** (`rope2d.py`): encodes (residue index, frame index) jointly in Q and K
- **AdaLN** (`adaln.py`): input normalization conditioned on diffusion time `t` and stride `Δt`
- **KV-cache**: context frames cached once at inference; only the target frame processed per denoising step

**StarIpaScore** (`gen_model/models/star_ipa.py`) non-invasively subclasses the upstream `IpaScore`, injecting ST attention blocks without modifying `extern/`.

**Two-stage training**:
1. Unconditional: score-match single frames → learns SE(3) backbone geometry
2. Conditional: multi-frame windows with Diffusion Forcing → learns temporal dynamics

---

## Citation

```bibtex
@article{yim2023se3,
  title={SE(3) diffusion model with application to protein backbone generation},
  author={Yim, Jason and others},
  journal={arXiv:2302.02277}, year={2023}
}
@article{jing2024mdgen,
  title={Generative modeling of protein ensemble from single sequence},
  author={Jing, Bowen and others},
  journal={arXiv:2407.01791}, year={2024}
}
@article{nikankin2022sinfusion,
  title={SinFusion: Training diffusion models on a single image or video},
  author={Nikankin, Yaniv and others},
  journal={arXiv:2211.11743}, year={2022}
}
```
