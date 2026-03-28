# STAR-MD: Spatio-Temporal Attention for Protein MD Trajectory Generation

This repository extends three prior works — **SE(3)-Diffusion**, **MDGen**, and **SinFusion** — to generate realistic protein molecular dynamics (MD) trajectories using SE(3) diffusion with spatio-temporal attention across frames.

---

## What this repo does differently

### Prior works

| Work | What it does |
|------|-------------|
| [SE(3)-Diffusion (Yim et al., 2023)](https://arxiv.org/abs/2302.02277) | SE(3) score-matching diffusion on protein backbones; single-structure generation |
| [MDGen (Jing et al., 2024)](https://arxiv.org/abs/2407.01791) | Extends SE(3)-Diffusion to multi-frame MD trajectories via conditional generation |
| [SinFusion (Nikankin et al., 2022)](https://arxiv.org/abs/2211.11743) | Single-video/image diffusion with temporal curriculum and SinFusion training pattern |

### Contributions of this repo

**1. STAR-MD: SpatioTemporalAttention module (`gen_model/models/star_attention.py`)**

Adds a cross-frame attention layer between IPA blocks so the model can reason across multiple MD frames jointly rather than frame-by-frame:

- **RoPE2D** (`gen_model/models/rope2d.py`): 2D rotary position embeddings encoding both residue index and frame index simultaneously, replacing 1D sinusoidal embeddings.
- **AdaLN** (`gen_model/models/adaln.py`): Adaptive Layer Normalization conditioned on diffusion time `t` and temporal gap `Δt = k·stride` (zero-initialized shift/scale, matching DiT-style conditioning).
- **Block-causal masking**: Attention is causal over frames (frame `i` attends only to frames `≤ i`) but fully connected within each frame, enabling autoregressive trajectory rollout.

**2. StarIpaScore (`gen_model/models/star_ipa.py`)**

A non-invasive subclass of the upstream `IpaScore` (from SE(3)-Diffusion) that injects `SpatioTemporalAttention` blocks between IPA trunk layers after `super().__init__()`. The upstream code is untouched — all STAR-MD logic lives in this extension file.

**3. SinFusion-style training (`gen_model/se3_diffusion_module.py`)**

Adopts SinFusion's training algorithm structure:

- `forward()` owns the complete pipeline: sample `t` → SE(3) forward marginal → model → score loss (analogous to SinFusion's `q_sample + MSE` in one method)
- `SE3Diffuser.forward_marginal()` is the exact analog of DDPM's `q_sample` — a one-shot closed-form marginal over SO(3) × R³
- `CosineAnnealingLR` (step-based, `eta_min = lr * 0.01`) replaces MultiStepLR
- `max_steps` instead of `max_epochs`
- Diffusion noise is applied in the Lightning module's `forward()`, not in the dataset — clean separation of data loading from diffusion

**4. Two-stage training with temporal curriculum**

Mirrors SinFusion's frame-difference curriculum:
- **Stage 1** (unconditional): Train on single frames with no temporal conditioning
- **Stage 2** (conditional): Start with `k ∈ {±1}`, growing to `k ∈ {±1, ±2, ±3}` every `grow_every` epochs via `TemporalCurriculumCallback`

**5. IPF spatial cropping**

Following MDGen, training uses Iterative Proportional Fitting (IPF)-balanced spatial crops so that each residue appears with uniform probability across the training distribution — a 3D analog of SinFusion's random spatial crops on images.

**6. LoRA fine-tuning support**

LoRA adapters can be applied to IPA attention projections (`linear_q`, `linear_kv`, `linear_out`) and STAR-MD projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) for parameter-efficient fine-tuning on new MD datasets.

---

## Repo structure

```
winter-gen-pproject/
├── extern/                          # Read-only upstream submodules
│   ├── se3_diffusion/               # Yim et al. 2023 (SE3-Diffusion)
│   ├── mdgen/                       # Jing et al. 2024 (MDGen)
│   └── sinfusion/                   # Nikankin et al. 2022 (SinFusion)
│
├── gen_model/
│   ├── path_setup.py                # Adds extern/ dirs to sys.path
│   │
│   ├── models/
│   │   ├── rope2d.py                # 2D RoPE for (residue, frame) positions
│   │   ├── adaln.py                 # AdaLN conditioned on (t, Δt)
│   │   ├── star_attention.py        # SpatioTemporalAttention with block-causal mask
│   │   ├── star_ipa.py              # StarIpaScore: non-invasive IpaScore extension
│   │   ├── score_network.py         # Shim → extern/se3_diffusion/model/score_network.py
│   │   └── lora.py                  # LoRA adapter injection
│   │
│   ├── diffusion/
│   │   ├── se3_diffuser.py          # Shim → extern/se3_diffusion/data/se3_diffuser.py
│   │   ├── so3_diffuser.py          # Shim → extern/se3_diffusion/data/so3_diffuser.py
│   │   └── r3_diffuser.py           # Shim → extern/se3_diffusion/data/r3_diffuser.py
│   │
│   ├── data/
│   │   └── dataset.py               # MDGenDataset + ConditionalMDGenDataset (modified from MDGen)
│   │
│   ├── se3_diffusion_module.py      # Lightning modules (SinFusion-style training)
│   ├── train_base.py                # Shared config helpers
│   ├── train_unconditional.py       # Stage 1 training
│   ├── train_conditional.py         # Stage 2 training with curriculum
│   │
│   └── splits/
│       ├── atlas.csv                # Protein metadata
│       └── frame_splits.csv         # Train/val/test frame assignments
│
└── checkpoints/
    ├── unconditional/               # Stage 1 checkpoints
    └── conditional/                 # Stage 2 checkpoints
```

---

## Usage

**Stage 1 — unconditional single-frame training:**
```bash
python gen_model/train_unconditional.py \
    --data_dir data \
    --max_steps 200000 \
    --batch_size 8 \
    --lr 1e-4
```

**Stage 2 — conditional multi-frame training with curriculum:**
```bash
python gen_model/train_conditional.py \
    --data_dir data \
    --max_steps 200000 \
    --batch_size 8 \
    --lr 1e-4 \
    --max_k 3 \
    --grow_every 10
```

**With LoRA (parameter-efficient fine-tuning):**
```bash
python gen_model/train_conditional.py \
    --data_dir data \
    --lora_r 8 \
    --lora_alpha 16
```

**With STAR-MD spatio-temporal attention:**
```bash
python gen_model/train_conditional.py \
    --data_dir data \
    --star_enabled   # (wire via default_model_conf star_enabled=True)
```

---

## Setup

```bash
git clone --recurse-submodules <this-repo>
pip install -r requirements.txt
```

The `extern/` submodules are read-only references to upstream repos. Do not commit changes into them.

---

## Citation

If you use this code, please cite the upstream works this builds on:

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
