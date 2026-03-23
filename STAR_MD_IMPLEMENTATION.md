# STAR-MD Implementation Plan

Implementation of the spatio-temporal SE(3) diffusion transformer from:
**"Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein Dynamics"**
on top of the existing IPA-based score network.

---

## Resolved Design Decisions

| # | Decision | Choice | Notes |
|---|----------|--------|-------|
| 1 | Training strategy | **A: Two-stage LoRA** | Unconditional pretraining → LoRA fine-tuning; chosen because training data is limited (single-protein regime). SpatioTemporalAttention weights always trained full-rank (not in unconditional checkpoint). |
| 2 | Frames per window (L) | **L=16** | Limited training data; 5× memory reduction vs. paper's L=80; covers a 1.6 ns window at Δt=0.1 ns. |
| 3 | Negative k for L>2 windows | **B: keep, sort ascending** | Window always sorted by frame_idx; target is last slot regardless of k sign |
| 4 | delta_t derivation | **B: sample Δt first, derive k** | Δt ~ LogUniform[0.01, 10] ns; k = round(Δt / ns_per_stored_frame), clamped to available trajectory length |

### On L

The paper fixes **L = 80** throughout training (no curriculum on L). At the paper's 1.25 ns/frame effective stride this covers a 100 ns window per sample. There is no warmup on L — training starts with 80-frame windows from epoch 0.

**Constraints for this codebase:**

- *Memory:* spatio-temporal attention operates on L×N tokens. At L=80, N=100 this is 8 000 tokens per attention layer per batch item. Feasible with gradient checkpointing; may require reducing batch size from the current default.
- *Trajectory length:* a sample of L frames at stride k requires `L × k` raw stored frames. At Δt=1 ns and ns_per_stored_frame=0.1 ns, k=10, needing 800 frames (~80 ns). At Δt=10 ns, k=100, needing 8 000 frames — beyond most ATLAS trajectories. Δt must be clamped per sample to `floor(total_frames / L) × ns_per_stored_frame`.
- *No k-curriculum interaction:* the paper has no curriculum on L or Δt. The existing `TemporalCurriculumCallback` (which grows max_k) becomes irrelevant if decision 4B is used — Δt is the primary training variable and k is derived from it.

**Decision: L=16.** Training data is limited (single-protein regime), making L=80 impractical. L=16 reduces spatio-temporal attention memory by 25× vs. the paper while preserving the mechanism. The `--num_frames` CLI default should be set to 16.

---

## What Changes and Why

The current model processes each frame independently: IPA computes SE(3)-equivariant attention within a single frame, and a vanilla `TransformerEncoder` does sequence-level attention within that frame. There is no coupling between frames during the forward pass.

STAR-MD adds **joint spatio-temporal attention** that attends across all `(residue, frame)` pairs simultaneously. This lets the model reason about how a residue's conformation at one frame relates to spatially nearby residues at other frames — the key capability for predicting long-horizon dynamics.

The changes are additive: IPA blocks stay, the sequence transformer stays (it operates within a single frame), and a new spatio-temporal attention module is appended after each IPA + sequence-transformer block.

---

## New Modules to Add

### 1. `RoPE2D` — 2D Rotary Position Embedding
**File:** `gen_model/models/rope2d.py` ✅ Already implemented

Encodes both the residue index `i` and the frame index `ℓ` into attention keys and queries. Allows the model to extrapolate to trajectory lengths not seen during training.

- Split the head dimension in half: first half encodes residue position, second half encodes frame position.
- Apply standard RoPE rotation to each half separately.
- Used inside the spatio-temporal attention module only (on Q and K; V is unchanged).

### 2. `SpatioTemporalAttention` — Joint (N×L)-token Attention
**New file:** `gen_model/models/star_attention.py`

Takes node embeddings from multiple frames and attends across all `N×L` tokens jointly.

**Inputs:**
- `s_frames`: `[B, L, N, c_s]` — node embeddings from `L` frames (L=1 at unconditional training time, L>1 for conditional)
- `frame_idx`: `[L]` — integer frame indices for RoPE
- `seq_idx`: `[B, N]` — residue indices for RoPE
- `mask`: `[B, L, N]` — combined mask
- `cond`: `[B, 2*D]` — conditioning vector for AdaLN, where `D = index_embed_size = 32` so `cond` has dim 64; carries both diffusion time `t` and log physical stride `log(Δt)`

**Forward:**
1. Apply AdaLN to input using `cond`
2. Flatten to `[B, L*N, c_s]` tokens; token ordering is `t = ℓ*N + i`
3. Apply 2D-RoPE to Q and K using `(seq_idx, frame_idx)` positions
4. Run multi-head self-attention (standard scaled dot-product)
5. Apply block-causal mask: tokens at frame `ℓ` can only attend to tokens at frame `ℓ' ≤ ℓ`
6. Apply padding mask from `mask` to zero-out padded residues
7. Unflatten back to `[B, L, N, c_s]`
8. Output projection with `"final"` init so the residual starts as identity

**Block-causal mask construction:**
```
attn_bias[b, h, (ℓ1*N + i), (ℓ2*N + j)] = -inf  if ℓ2 > ℓ1
                                           = 0     otherwise
```
This is a block lower-triangular mask of shape `[L*N, L*N]` built once from `L` and `N`, then added to attention logits before softmax. Combine with the padding mask: set `attn_bias[:, :, :, t] = -inf` for any token `t` that corresponds to a padded residue (i.e., `mask[b, ℓ, i] == 0`).

### 3. `AdaLN` — Adaptive Layer Normalization for Time Conditioning
**Add to `gen_model/models/modules.py`**

Used inside `SpatioTemporalAttention` to condition on diffusion timestep `t` and physical stride `Δt` jointly.

```
AdaLN(x, cond) = γ(cond) ⊙ LayerNorm(x) + β(cond)
```

where `γ` and `β` are each a two-layer MLP: `Linear(2*D, c_s) → SiLU → Linear(c_s, c_s)`. Initialize the **final** linear in each MLP with zero weights and unit/zero bias so that `γ=1, β=0` at the start of training (identity residual).

The conditioning vector is assembled in `ScoreNetwork.forward()` (see below) as:
```
cond = cat([get_timestep_embedding(t, D), get_index_embedding(log_delta_t_scaled, D)], dim=-1)
```
where `D = model_conf.embed.index_embed_size = 32` and `log_delta_t_scaled` is described in the `score_network.py` section.

---

## Files to Modify

### `gen_model/models/ipa_pytorch.py`

#### `IpaScore.__init__`

After the existing per-block modules, insert a `SpatioTemporalAttention` guarded by `star.enabled`:
```python
if getattr(model_conf, 'star', None) and model_conf.star.enabled:
    self.trunk[f'st_attn_{b}'] = SpatioTemporalAttention(
        c_s=ipa_conf.c_s,
        num_heads=model_conf.star.st_num_heads,
        causal=True,
    )
```

#### `IpaScore.forward` — signature change

Add two new keyword arguments:
```python
def forward(self, init_node_embed, edge_embed, input_feats,
            frame_idx=None, cond=None):
```

- `frame_idx`: `[L]` integer tensor of absolute frame positions; `None` or `[0]` for single-frame.
- `cond`: `[B, 2*D]` AdaLN conditioning vector; `None` for unconditional (st_attn receives zeros).

#### `IpaScore.forward` — multi-frame restructuring

The existing `forward` has **no per-frame loop** — it processes all residues in one frame via a `for b in range(num_blocks):` loop over blocks. The multi-frame extension works by treating the `L` frames as extra batch items, then applying cross-frame attention after each block.

**Step 1 — detect multi-frame and flatten:**
```python
# Determine L from input shape
input_rigids = input_feats['rigids_t']          # [B, L, N, 7] or [B, N, 7]
multi_frame = input_rigids.ndim == 4
B = input_rigids.shape[0]
L = input_rigids.shape[1] if multi_frame else 1
N = input_rigids.shape[-2]

if multi_frame:
    # Flatten B,L → BL for all per-frame tensors
    def flat(x):   return x.reshape(B*L, *x.shape[2:])
    def unflat(x): return x.reshape(B, L, *x.shape[1:])
    init_node_embed = flat(init_node_embed)     # [B*L, N, c_s]
    edge_embed      = flat(edge_embed)          # [B*L, N, N, c_z]
    node_mask       = flat(input_feats['res_mask'].float())  # [B*L, N]
    # rigids_t, fixed_mask, etc. flattened similarly
```

**Step 2 — run the existing per-block loop unchanged**, operating on the `[B*L, ...]` batch.

**Step 3 — insert cross-frame attention after each block:**
After `node_transition_b` and before `bb_update_b`, if `st_attn_{b}` exists:
```python
if multi_frame and f'st_attn_{b}' in self.trunk:
    seq_idx = input_feats['seq_idx']            # [B, N] — same for all frames
    st_mask = unflat(node_mask)                 # [B, L, N]
    node_embed_frames = unflat(node_embed)      # [B, L, N, c_s]
    node_embed_frames = self.trunk[f'st_attn_{b}'](
        node_embed_frames, frame_idx, seq_idx, st_mask, cond)
    node_embed = flat(node_embed_frames)        # back to [B*L, N, c_s]
```

**Step 4 — compute scores for all L frames:**
At the end of the forward, after all blocks, compute scores over the full `[B*L, ...]` output and reshape back to `[B, L, N, ...]`:
```python
# rot_score and trans_score over all B*L frames
rot_score = self.diffuser.calc_rot_score(
    init_rigids.get_rots(), curr_rigids.get_rots(), input_feats['t_flat'])   # [B*L, N, 3]
trans_score = self.diffuser.calc_trans_score(
    init_rigids.get_trans(), curr_rigids.get_trans(),
    input_feats['t_flat'][:, None, None], use_torch=True)                   # [B*L, N, 3]
_, psi_pred = self.torsion_pred(node_embed)                                  # [B*L, N, 2]

if multi_frame:
    rot_score   = unflat(rot_score)    # [B, L, N, 3]
    trans_score = unflat(trans_score)  # [B, L, N, 3]
    psi_pred    = unflat(psi_pred)     # [B, L, N, 2]
    curr_rigids_out = Rigid.from_tensor_7(unflat(curr_rigids.to_tensor_7()))  # [B, L, N]
```

`t_flat` is `input_feats['t'].unsqueeze(1).expand(B, L).reshape(B*L)` — the same τ broadcast to all frames. `Rigid` objects don't natively support reshape; use `.to_tensor_7()` → reshape → `Rigid.from_tensor_7()` throughout.

Each frame still gets its own `edge_embed`; `EdgeTransition` operates on the `[B*L, N, N, c_z]` flattened tensor unchanged.

For the single-frame unconditional case (`L=1`, `multi_frame=False`), no branch is taken and behavior is identical to the current code.

---

### `gen_model/models/score_network.py`

#### Building the `cond` vector

The AdaLN conditioning vector must be assembled from the diffusion time `t` and the physical stride `delta_t` before calling `IpaScore`. Add this to `ScoreNetwork.forward()`:

```python
D = self._model_conf.embed.index_embed_size  # 32

# t-embedding: reuse the existing sinusoidal embedder
t_emb = get_timestep_embedding(input_feats['t'], D)   # [B, D]

# delta_t-embedding: log-normalize to roughly [-1, 1] then embed
# LogUniform[0.01, 10] → log range [-4.61, 2.30], midpoint -1.15, half-width 3.46
delta_t = input_feats.get('delta_t', None)
if delta_t is not None:
    log_dt = torch.log(delta_t.float().clamp(min=1e-3))  # [B]
    log_dt_norm = (log_dt - (-1.155)) / 3.456            # zero-mean, unit-std approx
    dt_emb = get_index_embedding(log_dt_norm, embed_size=D)  # [B, D]
else:
    dt_emb = torch.zeros(input_feats['t'].shape[0], D,
                         device=input_feats['t'].device)

cond = torch.cat([t_emb, dt_emb], dim=-1)               # [B, 2*D = 64]
```

Pass `cond` through to `IpaScore`:
```python
model_out = self.score_model(node_embed, edge_embed, input_feats,
                             frame_idx=input_feats.get('frame_idx', None),
                             cond=cond)
```

#### Multi-frame Embedder calls

For `L > 1` frames, `Embedder` must be called **once per frame** because each frame has a different `sc_ca_t` (self-conditioning CA positions) and a different signed gap `k` relative to the target.

In `ScoreNetwork.forward()`, when `input_feats['rigids_t'].ndim == 4` (multi-frame batch):

```python
L = input_feats['rigids_t'].shape[1]
node_embeds = []
edge_embeds = []
for ℓ in range(L):
    # sc_ca_t for frame ℓ: use the CA positions of the PREVIOUS clean frame (ℓ-1),
    # or zeros for ℓ=0 (no prior context).
    sc_ca_ℓ = input_feats['sc_ca_t'][:, ℓ]        # [B, N, 3] pre-stored per frame
    k_ℓ     = input_feats.get('k_frames', None)    # [B, L] if provided; else None
    k_ℓ_val = k_ℓ[:, ℓ] if k_ℓ is not None else None

    n_emb, e_emb = self.embedding_layer(
        seq_idx=input_feats['seq_idx'],
        t=input_feats['t'],
        fixed_mask=fixed_mask,
        self_conditioning_ca=sc_ca_ℓ,
        k=k_ℓ_val,
    )
    node_embeds.append(n_emb)
    edge_embeds.append(e_emb)

init_node_embed = torch.stack(node_embeds, dim=1)   # [B, L, N, c_s]
init_edge_embed = torch.stack(edge_embeds, dim=1)   # [B, L, N, N, c_z]
```

For the single-frame case (`L=1`), the existing single `embedding_layer` call is unchanged.

#### Note on `k` vs `Δt`

The **existing** `k` field in `input_feats` is an integer frame gap used to index `temporal_embedder(k)` inside `Embedder` — it encodes "how many strides separate source from target" as a node feature. This is kept as-is.

`delta_t` is the **physical time** in nanoseconds corresponding to that gap. It is a separate scalar, computed by the dataset (see below) and used only for AdaLN conditioning inside `SpatioTemporalAttention`. Do not confuse or merge these two: `k` modulates node features; `delta_t` modulates the cross-frame attention.

---

### `gen_model/train_base.py`

**In `default_model_conf`:** add a `star` sub-config and update `target_modules` for LoRA:
```python
'star': {
    'enabled': True,
    'st_num_heads': 4,      # heads in spatio-temporal attention
    'causal': True,
},
```

Extend `lora.target_modules` to include the projections defined in `SpatioTemporalAttention`.
Name these projections consistently in `star_attention.py` as `q_proj`, `k_proj`, `v_proj`, `out_proj` so the following entry covers them:
```python
'target_modules': [
    'linear_q', 'linear_kv', 'linear_out',  # IPA
    'linear_1', 'linear_2',                  # MLP layers
    'q_proj', 'k_proj', 'v_proj', 'out_proj',  # SpatioTemporalAttention
],
```

**In `default_data_args`:** add `ns_per_stored_frame` (the physical time between consecutive stored frames in nanoseconds, **after** `frame_interval` subsampling):
```python
'ns_per_stored_frame': getattr(args, 'ns_per_stored_frame', 0.1),
```
This value must be set from trajectory metadata at data-preparation time.
`download_and_prep.py` already detects `ps_per_frame` from MDTraj; the conversion is:
```
ns_per_stored_frame = ps_per_frame * frame_interval / 1000
```
Pass it as a CLI argument `--ns_per_stored_frame` to training scripts.

---

### `gen_model/data/dataset.py`

#### `ConditionalMDGenDataset.__getitem__` — L-frame window

**Training objective (Diffusion Forcing):** This is NOT a "one context frame + one target frame" setup. Every frame in the window is simultaneously a denoising target AND a context for later frames, following the Diffusion Forcing approach. All L frames receive the same small noise level τ ~ U[0, 0.1] (making them approximately clean), and loss is computed at all L positions in a single forward pass. The block-causal mask provides the causal structure: frame ℓ attends to frames 0..ℓ-1 when computing its denoising score, so the model learns `p(x_ℓ | x_{<ℓ})` for all ℓ simultaneously.

**L and inference:** L is the training window size (paper: L=80 for the 100 ns benchmark). At inference, the rolling context grows beyond L as more frames are generated — RoPE2D is specifically designed to extrapolate to history lengths not seen during training.

**Sampling `delta_t` and deriving the frame stride (Decision 4B):**

```python
# 1. Sample physical stride from the paper's distribution
delta_t_ns = np.exp(np.random.uniform(np.log(0.01), np.log(10.0)))  # LogUniform[0.01, 10] ns

# 2. Convert to raw-frame stride
k_raw = int(round(delta_t_ns / self.args.ns_per_stored_frame))
k_raw = max(1, k_raw)  # minimum 1 stored frame apart

# 3. Clamp to what the trajectory can support for L frames
max_k_available = (total_frames - 1) // (L - 1)
k = min(k_raw, max_k_available)

# 4. Actual physical stride used (quantized to stored frame grid)
delta_t_actual = k * self.args.ns_per_stored_frame  # what AdaLN receives
```

**Handling negative k (Decision 3B):**

```python
# Sample direction: ±1 with equal probability (data augmentation)
direction = np.random.choice([-1, 1])

# Build L frame indices, evenly spaced by k in the chosen direction, sorted ascending
raw_indices = [frame_start + direction * i * k for i in range(L)]
frame_idx = np.array(sorted(raw_indices))   # [L], always ascending
```

The block-causal mask uses position in the sorted window. The model predicts each frame ℓ given sorted frames 0..ℓ-1, regardless of physical time direction.

**Frame selection and validation:** `frame_start` must be chosen so all `raw_indices` are within `[0, total_frames - 1]`. `_add_frames_for_split` must require `(L-1) * k_max` frames on both sides of every valid start, where `k_max = floor(max_trajectory_frames / (L-1))`.

**Noise application (same τ for all frames):**
Sample one τ ~ U[0, 0.1] per training example and apply `_apply_diffusion(clean_rigids_ℓ, tau, mask)` to every frame in the window with the same τ. This is consistent with the Diffusion Forcing training scheme: all frames are approximately clean, and the model learns to denoise any of them given earlier frames.

**Self-conditioning (`sc_ca_t`) per frame:**
- `sc_ca_t[ℓ=0]` = zeros (no prior context)
- `sc_ca_t[ℓ>0]` = clean CA positions of frame `ℓ-1` (centred + scaled)

Store as `'sc_ca_t': tensor [L, N, 3]`.

**The sample should return:**
```python
'rigids_t':              [L, N, 7]   # all L frames noised at the same τ ~ U[0, 0.1]
'rigids_0':              [L, N, 7]   # clean ground-truth for all L frames
'rot_score':             [L, N, 3]   # denoising score for all L frames at noise level τ
'trans_score':           [L, N, 3]   # denoising score for all L frames at noise level τ
'rot_score_scaling':     scalar      # shared across all frames (same τ)
'trans_score_scaling':   scalar
'sc_ca_t':               [L, N, 3]   # per-frame self-conditioning CA positions
'frame_idx':             [L]         # sorted absolute frame indices in trajectory
'delta_t':               scalar      # actual physical stride (quantized), in ns
't':                     scalar      # the shared noise level τ used for all frames
# All existing per-protein fields (aatype, seq_idx, res_mask, fixed_mask, etc.)
# remain [N] or scalar — same for all L frames.
```

Note: `'k'` and `'k_frames'` are dropped. `delta_t` is the only temporal conditioning signal. `temporal_embedder(k)` in `Embedder` is no longer used for multi-frame training.

**Collation:** `L` is fixed per training run (CLI arg `--num_frames`, default 80). Default PyTorch collation produces `[B, L, N, ...]` without a custom collate_fn.

For unconditional training, `L=1` and no change is needed.

---

### `gen_model/train_base.py` — `SE3BaseModule._compute_loss`

For multi-frame batches, `rot_score` and `trans_score` are `[B, L, N, 3]` and loss is computed over **all L frames**, not just the last. `pred['rot_score']` and `pred['trans_score']` are also `[B, L, N, 3]` (returned by `IpaScore` for the full window).

```python
if batch['rot_score'].ndim == 4:       # multi-frame: [B, L, N, 3]
    rot_score   = batch['rot_score']           # [B, L, N, 3]
    trans_score = batch['trans_score']
    # res_mask is [B, N] (same crop for all frames); expand for broadcasting
    mask = batch['res_mask'].float()[:, None, :].expand_as(rot_score[..., 0])  # [B, L, N]
    rot_scaling   = batch['rot_score_scaling'][:, None, None, None]    # [B, 1, 1, 1]
    trans_scaling = batch['trans_score_scaling'][:, None, None, None]
    n_visible = mask.sum() + 1e-8
else:
    rot_score   = batch['rot_score']
    trans_score = batch['trans_score']
    mask = batch['res_mask'].float()
    rot_scaling   = batch['rot_score_scaling'][:, None, None]
    trans_scaling = batch['trans_score_scaling'][:, None, None]
    n_visible = mask.sum() + 1e-8
```

The L×N tokens each contribute equally to the loss. This gives L× more training signal per trajectory window than the single-target approach, which is the main efficiency argument for the Diffusion Forcing training scheme.

`IpaScore.forward` must correspondingly return scores for all L frames (not just frame L-1). Remove the "extract target frame only" step described in the IpaScore section above — instead, return `rot_score [B, L, N, 3]` and `trans_score [B, L, N, 3]` from the full `[B*L, ...]` flattened output. Torsion prediction (`psi`) can similarly be computed for all frames or only the last (psi loss contribution is minor; all-frame is cleaner).

---

### `gen_model/train_conditional.py`

- Replace the current source+target curriculum with an `L`-frame window dataset (new CLI arg `--num_frames`, default 80).
- `delta_t` is returned by the dataset; `ScoreNetwork.forward()` reads it from `input_feats['delta_t']` automatically.
- Remove `TemporalCurriculumCallback` — the k-curriculum is superseded by Δt sampling.
- Loss is computed over **all L frames** via `_compute_loss` changes above.

---

## Training Changes

### Unconditional training (`train_unconditional.py`)
No changes. `L=1`, spatio-temporal attention is a no-op.

### Conditional training (`train_conditional.py`)
- Replace the current source+target curriculum with an `L`-frame window dataset
- All L frames are noised at the same small τ ~ U[0, 0.1]; loss is computed at all L positions (Diffusion Forcing)
- `delta_t` sampled first (LogUniform[0.01, 10] ns); frame stride `k` derived from it
- Block-causal masking ensures each frame is conditioned only on earlier frames in the sorted window
- Remove `TemporalCurriculumCallback`; remove `current_max_k` logic

---

## Inference / Autoregressive Rollout

**New file:** `gen_model/inference_conditional.py`

The conditional model generates trajectories autoregressively:

```
for ℓ = 1, 2, ..., T_total:
    1. Collect history: frames 0…ℓ-1 (clean generated frames, lightly noised)
    2. Run reverse diffusion (N_steps denoising steps) for frame ℓ,
       conditioned on history via block-causal SpatioTemporalAttention
    3. Apply context noise τ ~ U[0, 0.1] to the newly generated frame ℓ
    4. Append noised frame ℓ to history for next iteration
```

**Critical: context noise at inference.** After each frame is denoised to `x̂_ℓ`, apply a small forward diffusion with `τ ~ U[0, 0.1]` before storing it as history. This matches the training distribution (context frames were always lightly noised during training). Skipping this step causes distribution shift and error accumulation over long rollouts.

**`delta_t` at inference:** fixed by the user as the desired physical stride (e.g., `0.1` ns between frames). Pass as a scalar to `ScoreNetwork` for every denoising step.

**KV cache (efficiency):** During autoregressive rollout, the key and value tensors for all context frames in `SpatioTemporalAttention` are recomputed from scratch each step. To avoid O(T²) cost, cache K and V for context tokens:
- On the first denoising step for frame `ℓ`, compute and store K/V for all context frames 0…ℓ-1.
- On subsequent denoising steps for the same frame `ℓ`, reuse the cached context K/V and only compute K/V for the (unchanged) context; the target token K/V change with each denoising step.
- This reduces the per-frame cost from O(N²L²) to O(N²L) amortised.

KV caching is optional for correctness but important for scaling beyond ~20 frames.

---

## What Stays the Same

- All IPA machinery (`InvariantPointAttention`, `BackboneUpdate`, `EdgeTransition`, `StructureModuleTransition`)
- Per-frame sequence transformer (`seq_tfmr`) — still runs within each frame before `st_attn`
- Per-frame pair features (`edge_embed`) — `EdgeTransition` still operates per-frame on the `[B*L, ...]` flattened batch
- All initialization conventions (LeCun default, `"final"` for residual outputs)
- SE(3) diffusion math (`SE3Diffuser`, `SO3Diffuser`, `R3Diffuser`)
- Inference scripts for single-frame generation (unconditional) unchanged
- Checkpoint compatibility for existing weights (new modules initialize to zero residual)
- `temporal_embedder(k)` in `Embedder` — provides per-frame node feature for signed gap; unchanged

---

## Rollout Order

1. ~~`rope2d.py`~~ — already done
2. ~~`AdaLN` in `modules.py`~~ — self-contained ✅ Done
3. `star_attention.py` using `RoPE2D` and `AdaLN`
4. Wire `SpatioTemporalAttention` into `IpaScore` behind `star.enabled` flag; update signature to accept `frame_idx` and `cond`; add `star` sub-config to `default_model_conf`; extend LoRA `target_modules`
5. Update `ScoreNetwork.forward()`: build `cond` vector from `t` and `delta_t`; call `Embedder` per-frame for multi-frame inputs; pass `frame_idx` and `cond` to `IpaScore`
6. Update `SE3BaseModule._compute_loss` to index target frame when batch has L dimension
7. Update `ConditionalMDGenDataset` to return `L`-frame windows with per-frame `sc_ca_t`, `frame_idx`, `delta_t`, and light context noise
8. Update `default_data_args` to accept `ns_per_stored_frame`; update `train_conditional.py` with `num_frames` arg
9. Write `inference_conditional.py` with autoregressive rollout + context noise injection
10. Validate: unconditional training with `star.enabled=False` must reproduce baseline loss curve

---

## Intentional Differences from the STAR-MD Paper

The following design choices diverge deliberately from the paper. They are not gaps — each is a reasoned adaptation to this codebase's constraints and goals.

### 1. Single-protein / small-dataset focus

**Paper:** Trains on the full ATLAS dataset — hundreds of diverse proteins with multiple MD replicas each. The model learns generalizable dynamics across folds and sequences.

**This codebase:** Designed to train on one protein's trajectory (or a small curated set). `MDGenDataset` indexes frames from whatever trajectories are in `data_dir` and the split CSV, with no assumption of diversity. There is no multi-protein balancing or protein-level sampling weight; every frame in the split is equally likely.

**Implication for STAR-MD additions:** `ns_per_stored_frame` is a single global value because all trajectories in a run are assumed to share the same physical timestep. If multi-protein training is desired in the future, this becomes a per-trajectory metadata field.

---

### 2. Spatial cropping with IPF-balanced seed selection

**Paper:** Operates on the full residue sequence; no spatial masking during training.

**This codebase:** `MDGenDataset.__getitem__` applies a spatial crop to `crop_ratio` (default 0.95) of residues at each training step. A seed residue is selected according to pre-computed Iterative Proportional Fitting (IPF) weights (`_compute_balanced_weights`), which are calibrated so that each residue appears in approximately the same number of crops across training. The `keep_len = int(L * crop_ratio)` nearest residues (by Cα distance in the reference frame) are kept; the rest are zeroed out via `res_mask`.

**Purpose:** Memory and compute reduction for longer proteins; implicit data augmentation by presenting different spatial subsets at each step; IPF weighting corrects the bias that central residues would otherwise be overrepresented.

**Interaction with STAR-MD:** The `res_mask` used for cropping is per-frame. For multi-frame batches, the same spatial crop (same seed, same `keep_indices`) must be applied to **all L frames** consistently — a different crop per frame would break the cross-frame correspondence that `SpatioTemporalAttention` relies on. The dataset change in step 7 must propagate the crop mask from the source frame to all context and target frames.

---

### 3. Soft local attention masking in IPA and sequence transformer

**Paper:** Uses fully global IPA attention and global sequence-transformer attention across all residue pairs.

**This codebase:** Two optional soft Gaussian masks restrict attention range:

- `local_attn_sigma` (Ångströms): added as `-(2·dist/σ)²` to IPA attention logits before softmax. Attention to a residue at distance `σ` falls to ~37%; at `2σ` to ~2%. Recomputed every block as frames are refined.
- `seq_tfmr_sigma` (residues): same Gaussian bias applied to the sequence transformer's attention logits, using sequence distance `|i - j|` instead of 3D distance. Computed once before the block loop.

**Purpose:** Reduces quadratic memory/compute for large proteins; acts as an inductive bias that nearby residues are more relevant. Values of 12–20 Å for IPA and 16–32 residues for seq_tfmr are typical.

**Interaction with STAR-MD:** These masks apply only within-frame (IPA and seq_tfmr are per-frame). `SpatioTemporalAttention` uses its own block-causal mask and does not inherit these biases — it always attends globally across all `L*N` tokens.

---

### 4. Two-stage training: unconditional pretraining + LoRA conditional fine-tuning

**Paper:** Trains a single model end-to-end on the conditional objective (all parameters updated jointly from scratch or from a pretrained structure predictor).

**This codebase:** Trains in two stages:
1. `train_unconditional.py` — trains the full `ScoreNetwork` (IPA trunk + embedder) on the single-frame denoising objective with no temporal context.
2. `train_conditional.py` — loads the unconditional checkpoint and applies LoRA to a subset of linear layers (`linear_q`, `linear_kv`, `linear_out`, `linear_1`, `linear_2`). Only the LoRA parameters are updated; the base IPA weights are frozen.

**Purpose:** The unconditional model provides a strong geometric prior for single-frame structure quality. Fine-tuning only the LoRA parameters keeps the conditional model close to this prior and prevents catastrophic forgetting when the conditional training data is limited (single-protein regime).

**For STAR-MD:** LoRA `target_modules` must be extended to include `SpatioTemporalAttention`'s projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`). The new `st_attn` modules are always trained from scratch (not loaded from the unconditional checkpoint, since they don't exist there), so they are effectively full-rank even when LoRA is enabled for the rest of the network. This is intentional: cross-frame attention is entirely new capacity that must be learned from scratch.

---

### 5. Temporal curriculum for k

**Paper:** No curriculum. Trains with the full Δt ~ LogUniform[0.01, 10] ns distribution and fixed L=80 from epoch 0.

**This codebase:** `TemporalCurriculumCallback` starts with `current_max_k = 1` (only `k ∈ {-1, +1}`) and increments by 1 every `grow_every` epochs until reaching `max_k`.

**For STAR-MD (Decision 4B):** The k-curriculum is superseded. `delta_t` is now sampled from LogUniform[0.01, 10] ns directly; `k` is derived from it and clamped to trajectory length. `TemporalCurriculumCallback` becomes a no-op for multi-frame training and can be removed from `train_conditional.py`. The frame-index builder (`_add_frames_for_split`) must instead require `L × k_max` frames on both sides of each valid start, where `k_max = floor(max_trajectory_frames / L)`.

If instability is observed early in training, a lightweight curriculum on the LogUniform upper bound (e.g., clamp to 1 ns for the first 20 epochs, then release to 10 ns) is a reasonable fallback, but the paper does not use one.

---

### 6. Bidirectional temporal gap (negative k)

**Paper:** Strictly forward/autoregressive — context always precedes the target in physical time.

**This codebase:** `ConditionalMDGenDataset` samples `k` from `{-current_max_k, …, -1, +1, …, +current_max_k}`, so the model also trains on predicting a frame in the **past** from a later source frame. The signed k is passed through to `Embedder.temporal_embedder(k)` which distinguishes positive and negative values via its sinusoidal encoding.

**Purpose:** Data augmentation — doubles the number of (source, target) pairs available from each trajectory without requiring additional data. The underlying SE(3) diffusion objective is symmetric with respect to which direction "time" runs; the model is just learning the conditional distribution `p(x_target | x_source)` regardless of sign.

**For STAR-MD:** The block-causal mask inside `SpatioTemporalAttention` enforces a strict ordering of the L frames by their `frame_idx` values. If a window with `k < 0` is included (target frame temporally before context frames), `frame_idx` must be sorted ascending before building the causal mask — the causal direction is defined by `frame_idx` order, not by the sampling sign of `k`. In practice, sort the L frames by `frame_idx` and assign `frame_idx = [0, 1, …, L-1]` so that the target is always last, regardless of the original sign of k. This is the correct interpretation: the model predicts the final frame given earlier ones, but "earlier" means earlier in the sorted window, not necessarily earlier in physical time.

---

### 7. Global coordinate centering and per-dataset scale normalization

**Paper:** Does not describe explicit coordinate normalization beyond the SE(3) frame representation.

**This codebase:** At dataset construction time, `_compute_coord_scale()` computes `1 / std(all_CA_coords)` from a sample of training frames. Every frame's translations are then centered at the CA centroid and multiplied by `coord_scale` before being passed to the model. The same `coord_scale` is applied to all frames in a run; the val dataset copies it from the train dataset (`val_dataset.coord_scale = float(train_dataset.coord_scale)`).

**Purpose:** Normalizes translation magnitudes to a consistent scale regardless of protein size. Without this, larger proteins (wider spatial extent) would dominate the translation loss.

**Interaction with STAR-MD:** The scaling is already applied in the dataset before any STAR-MD logic runs. `delta_t_ns` is in physical nanoseconds and is not affected by `coord_scale`. The `cond` vector carries raw `log(delta_t_ns)`, so the physical time conditioning is independent of the coordinate normalization.

---

### 8. Source CA as self-conditioning (`sc_ca_t`)

**Paper:** Does not describe self-conditioning in the same sense. The conditioning on prior frames flows entirely through the spatio-temporal attention mechanism.

**This codebase:** The `Embedder` computes a pairwise CA-distance distogram from `sc_ca_t` and appends it to the edge features `[B, N, N, D_edge]`. In the unconditional model this is zeros (no self-conditioning). In the conditional model it is set to the source frame's CA positions, giving the model an explicit spatial hint via pair features before any attention is computed.

**For STAR-MD:** `sc_ca_t` is set per-frame as described in the dataset section — frame `ℓ` uses the clean CA positions of frame `ℓ-1`. This preserves the existing conditioning mechanism while extending it to multiple frames. The spatio-temporal attention then augments this with learned cross-frame coupling; the two mechanisms are complementary rather than redundant.

---

### 9. Replicate-based train/val/test split

**Paper:** Trains on the full ATLAS dataset; split strategy not the focus of the paper.

**Current codebase:** `frame_splits.csv` defines temporal boundaries (`train_early_end`, `train_end`, `val_end`) within each trajectory. All three replicates appear in every split. This means train/val/test are contiguous chunks of the same trajectories — frames within one trajectory are highly correlated, so the val/test boundary is somewhat arbitrary and ~500 frames are silently excluded from training (the `train_early` buffer used only for IPF reference).

**This codebase (new design):** Split at the replicate level — each replicate is assigned entirely to one partition:
- `R1` → training
- `R2` → validation
- `R3` → test
- `_latent` variants follow their base replicate (e.g. `R1_latent` → training)

**Why this is better:**
- Different replicates start from different initial velocities and explore the conformational space independently — they are genuinely held-out dynamics, not just later timepoints of the same trajectory.
- All ~10,000 frames of R1 are available for training instead of ~5,700 under the temporal split.
- Eliminates the 500-frame excluded buffer and the arbitrary mid-trajectory train/val boundary.
- For STAR-MD's goal (long-horizon dynamics), evaluating on a held-out replicate tests whether the model reproduces the correct conformational ensemble — a harder and more informative metric.

**Implementation:** Three separate CSV files, each using the full trajectory (`train_end = total_frames`, no `val_end` or `train_early_end` needed):

`gen_model/splits/train_splits.csv`:
```
name,train_end,total_frames
1a62_A_R1,10001,10001
4o66_C_R1,10001,10001
4o66_C_R1_latent,10001,10001
```

`gen_model/splits/val_splits.csv`:
```
name,train_end,total_frames
1a62_A_R2,10001,10001
4o66_C_R2,10001,10001
4o66_C_R2_latent,10001,10001
```

`gen_model/splits/test_splits.csv`:
```
name,train_end,total_frames
1a62_A_R3,10001,10001
4o66_C_R3,10001,10001
4o66_C_R3_latent,10001,10001
```

The dataset already supports this via separate `train_split` and `val_split` CLI args — `MDGenDataset` reads `args.train_split` for train mode and `args.val_split` for val mode. Pass `--train_split splits/train_splits.csv --val_split splits/val_splits.csv` to training scripts. For test-time evaluation, pass `splits/test_splits.csv` directly.

**Coord scale:** `_compute_coord_scale()` is computed from R1 (training data) and copied to the val dataset as before (`val_dataset.coord_scale = float(train_dataset.coord_scale)`). No change needed here.

**`train_early_end` buffer:** No longer needed. The IPF balanced-weight reference frame (`ref_frame_idx`) should default to frame 0 of R1 since there is no longer a distinct early-training segment. Verify that `_compute_balanced_weights` and `_compute_coord_scale` still sample from the correct dataset when `train_early_end` is absent from the CSV.
