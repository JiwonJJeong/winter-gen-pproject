# Inspiring Technical Works

Running catalog of works that inform STAR-MD design choices. Each entry is a
brief note on **what the work does** and **why it matters here** — not an
abstract. Add to this as new ideas come up; remove entries that turn out to be
dead ends.

Format:
- **Title** (Authors, Year / Venue) — one-line takeaway. *Why for this project:* …

---

## Block-AR & Diffusion Forcing

- **Diffusion Forcing** (Chen et al., NeurIPS 2024) — per-token independent
  noise levels with causal masking; at inference, finalized tokens sit at t=0
  while new tokens denoise from t≈1.
  *Why for this project:* foundation of the per-frame t branch in
  `_apply_se3_noise` and the synchronous block-AR rollout. The model trains
  on mixed-noise contexts so any per-frame t schedule is in-distribution.

- **Rolling Diffusion** (Ruhe et al., ICML 2024) — pyramidal noise schedule:
  frame ℓ in the active window sits at t_ℓ = ℓ·Δt, oldest finalizes first,
  new pure-noise frame appended at the back.
  *Why:* the natural follow-up to our synchronous block schedule. Removes
  block-boundary artifacts since there are no boundaries — the window slides
  continuously.

- **CausVid** (Yin et al., NVIDIA, 2024) — distill a bidirectional teacher
  into a few-step causal student.
  *Why:* a path to fast block-AR inference once the bidirectional Wan-style
  ceiling is established. Not relevant until we have a strong teacher.

- **Self-Forcing** (2024) — train AR diffusion with its own noisy past as
  context (scheduled sampling for diffusion) to close the exposure-bias gap.
  *Why:* directly addresses the drift problem in the existing K=1 rollout.
  Worth trying if block-AR alone doesn't fix long-horizon degradation.

- **MAGI-1** (Sand AI, 2025) — production block-AR video: chunk denoising
  with bidirectional intra-chunk attention, AR across chunks.
  *Why:* validates K ≈ 1s of video as a practical block size. Reference for
  block-boundary handling (history conditioning, overlap windows).

- **FIFO-Diffusion** (Kim et al., 2024) — extend a fixed-length video
  diffusion model to arbitrary length by rotating noise levels through a
  queue.
  *Why:* alternative to KV-cache rolling-window inference. Trades off
  cache simplicity for compatibility with off-the-shelf bidirectional models.

## Bidirectional video diffusion (the ceiling we're moving away from)

- **Wan 2.x** (Alibaba, 2024–2025) — open frontier bidirectional video DiT,
  full 3D attention with sparse / windowed factorizations at scale.
  *Why:* the quality target. Block-AR's job is to recover most of Wan's
  intra-clip coherence while gaining streaming and arbitrary length.

- **HunyuanVideo** (Tencent, 2024) — 13B parameter open video DiT, full
  spatio-temporal attention.
  *Why:* peer reference for "what good motion coherence looks like" without
  AR. Useful as a sanity benchmark on shared evals.

- **CogVideoX** (THUDM, 2024) — DiT-based open video model with expert
  transformer architecture.
  *Why:* open weights, useful for ablations on attention structure.

- **NOVA** (BAAI, 2024) — non-quantized autoregressive video model
  (continuous tokens, set-by-set prediction).
  *Why:* an alternative AR formulation that sidesteps the per-frame
  diffusion cost. Worth understanding if we ever want to drop diffusion
  entirely on the temporal axis.

## Single-instance / single-trajectory diffusion

- **SinFusion** (Nikankin et al., 2022) — single-video/single-image
  diffusion via virtual epochs, curriculum, spatial augmentation.
  https://arxiv.org/abs/2211.11743
  *Why:* the anti-overfitting playbook this project explicitly adapts.
  Crops, virtual epochs, and δt curriculum are direct ports.

- **SinGAN** (Shaham et al., ICCV 2019) — predecessor: GAN trained on
  patches of a single natural image.
  *Why:* historical context for SinFusion; informs why patch-level
  augmentation matters in single-instance training.

## Protein MD / SE(3) diffusion

- **SE(3)-Diffusion** (Yim et al., 2023) — score-matching diffusion on
  SE(3) for protein backbones.
  https://arxiv.org/abs/2302.02277
  *Why:* the backbone of this codebase (`extern/se3_diffusion`). All
  rotation/translation noise math comes from here.

- **MDGen** (Jing et al., 2024) — extends SE(3)-Diffusion to multi-frame
  MD trajectories via conditional generation; uses EMA, virtual epochs.
  https://arxiv.org/abs/2407.01791
  *Why:* the multi-frame extension this project builds on. EMA pattern
  in `se3_diffusion_module.py` is a direct port.

- **STAR-MD** (Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon
  Protein Dynamics) — the architecture being implemented in this repo.
  *Why:* defines the joint (N×L)-token attention with 2D-RoPE, AdaLN,
  block-causal masking, IPA → ST attention → coord prediction.
  Implementation lives in `gen_model/models/star_*.py`.

- **AlphaFold 3** (Abramson et al., Nature 2024) — diffusion-based
  generation of all-atom protein structures.
  *Why:* not directly used, but its conditioning patterns (per-token
  conditioning vectors) are the conceptual ancestor of our AdaLN cond.

## Architectural building blocks

- **DiT** (Peebles & Xie, ICCV 2023) — diffusion transformers with
  AdaLN-Zero conditioning.
  *Why:* the zero-init residual pattern in `AdaLN` and `out_proj` follows
  AdaLN-Zero exactly. The "(1 + γ)" structural one is from this paper.

- **2D-RoPE** (extension of Su et al., RoFormer 2021) — rotary position
  embedding split per axis for joint (residue, frame) encoding.
  *Why:* core of `rope2d.py`. Generalizes naturally to per-frame t when
  we add temporal-only relative encoding later.

- **Self-conditioning** (Chen et al., 2022, "Analog Bits") — recycle
  predicted x_0 as input conditioning each diffusion step.
  *Why:* the `sc_ca_t` field in the dataset is a Markov-1 variant. Worth
  revisiting whether full self-conditioning (predicted-x_0 feedback)
  improves quality.
