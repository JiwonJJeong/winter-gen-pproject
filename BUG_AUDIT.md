# Bug Audit Map — STAR-MD

## Category 1 — Diffusion Math ✅ AUDITED

Scope: how our code **uses** extern diffusion APIs (not extern internals).

### Results

| File | Finding |
|---|---|
| `gen_model/diffusion/se3_diffuser.py` | Pure re-export shim. No bugs. |
| `gen_model/diffusion/so3_diffuser.py` | Pure re-export shim. No bugs. |
| `gen_model/diffusion/r3_diffuser.py` | Pure re-export shim. No bugs. |
| `gen_model/path_setup.py` | Both patches (`rot_to_quat`, `torch_score`) are correct. Module-level Python caching ensures each runs exactly once. No bugs. |
| `gen_model/se3_diffusion_module.py` | `forward_marginal` API consumed correctly (keys, types, float32 cast via `_to_tensor`). **Code smell (line 195-196):** `diff` variable referenced outside the inner `for l in range(L)` loop to record scalings — works because `score_scaling(t)` is a pure function of `t_i`, but silently wrong if scaling ever becomes frame-dependent. No actionable bug now. |

### Verdict
No bugs in Category 1. All extern API calls are correct. Code smell in `_apply_se3_noise` multi-frame scaling reference is noted but not a defect.

---

## Category 2 — STAR Model Architecture ✅ AUDITED

### Results

| File | Finding |
|---|---|
| `gen_model/models/adaln.py` | Clean. Zero-init of γ/β final layers is correct. Broadcast via `unsqueeze(1)` loop is correct for 3D and 4D inputs. |
| `gen_model/models/rope2d.py` | Clean. `seq_pos`/`frame_pos` construction is correct. Rotate-half trick is correct. |
| `gen_model/models/star_attention.py` | Clean. Block-causal mask via `repeat_interleave` is correct. Spatial Gaussian bias shape math is correct. |
| `gen_model/models/star_ipa.py` | Clean. `scale_rigids`/`unscale_rigids` inherited correctly from upstream `IpaScore`. `ca_pos` unscaling before spatial bias is correct. Note: `init_node_embed` is masked then `node_embed = init_node_embed * mask` double-masks, but binary idempotency means no numerical effect. |
| `gen_model/models/star_score_network.py` | See bug below. |
| `gen_model/data/dataset.py` | **BUG FIXED** — `frame_idx` was absolute trajectory positions; see below. |

### Bug Fixed: `frame_idx` train-inference mismatch (Medium)

**Root cause**: `ConditionalMDGenDataset.__getitem__` set `frame_idx = sorted(raw_trajectory_indices)` — absolute positions like `[1000, 1005, ..., 1075]` for stride k=5. Since RoPE attention scores depend only on *relative* position differences, training adjacent frames had relative diff = k (e.g., 5) while inference adjacent frames (`frame_abs = 0, 1, 2, ...`) always had relative diff = 1. Complete temporal RoPE mismatch between training and inference.

**Secondary effect**: `star_score_network.py` takes `frame_idx[0]` from the collated `[B, L]` batch tensor (comment: "frame indices are the same across the batch"). With absolute trajectory positions this was also wrong: each batch item had different absolute positions (different `frame_start`, direction, k), so items 1..B-1 got the wrong temporal position embeddings.

**Fix** (`gen_model/data/dataset.py`): changed to `torch.arange(L, dtype=torch.long)` — relative window positions `[0, 1, ..., L-1]`. Now:
- Relative position difference between adjacent training frames = 1, matching inference.
- All batch items share the same `frame_idx`, so `frame_idx[0]` in `star_score_network.py` is now correct.
- Physical stride information is still communicated to the model via `delta_t` / AdaLN conditioning.

---

## Category 3 — Loss & Training Loop ✅ AUDITED

### Results

| File | Finding |
|---|---|
| `gen_model/se3_diffusion_module.py` | Clean. Rot-loss `n_rot` normalization correctly counts only (visible × t<threshold) tokens. Trans/psi shapes correct for both single- and multi-frame. Aux losses correctly gated by `'atom37' in pred` (absent for multi-frame). EMA `on_train_batch_end` correctly fires post-optimizer-step. |
| `gen_model/models/lora.py` | **BUG FIXED** — see below. Base weights frozen correctly; `freeze_non_lora` walk is correct. |

### Bug Fixed: LoRA forward materializes full weight matrix (Minor/Efficiency)

**Root cause**: `LoRALinear.forward` computed `F.linear(x, self.lora_B @ self.lora_A)`, which first materializes the full `(d_out × d_in)` product matrix at every forward call. This is O(d²) memory and compute — the exact overhead LoRA is designed to avoid.

**Fix** (`gen_model/models/lora.py`): replaced with two sequential small matmuls `F.linear(F.linear(x, self.lora_A), self.lora_B)`, which is O(r·d) and never allocates a full-rank intermediate. Mathematically identical.

---

## Category 4 — Data Pipeline
Frame indexing, stride, and cropping bugs cause subtle distributional errors.

| File | What to check |
|---|---|
| `gen_model/data/dataset.py` | `ConditionalMDGenDataset` window boundaries; `delta_t` LogUniform sampling; IPF-balanced crop correctness |
| `gen_model/data/geometry.py` | atom14↔atom37 conversions; torsion angle computation edge cases |

---

## Category 5 — Inference
Autoregressive rollout bugs produce drifting/degenerate trajectories.

| File | What to check |
|---|---|
| `gen_model/inference_conditional.py` | KV-cache invalidation; frame context window shifting; reverse SDE step correctness |
| `gen_model/inference_unconditional.py` | Sampling consistency with training noise schedule |

---

## Category 6 — Tests (Verify Coverage)
Check that tests actually catch the bugs they claim to cover.

| File | What to check |
|---|---|
| `tests/test_balanced_masking.py` | IPF crop probability uniformity — a stat test, not just a shape check |
| `tests/test_scaling.py` | Round-trip `scale → unscale` identity check |
| `tests/test_predictor_dataset.py` | Window stride and boundary conditions |

---

## Recommended Audit Order

1. **Category 1** — dtype/numerical bugs in diffusion math are load-bearing and affect everything downstream
2. **Category 2** — STAR attention/RoPE/AdaLN stack is the novel code with the most surface area for shape bugs
3. **Category 3** — verify loss terms are correctly normalized and aux losses match the upstream spec
4. **Category 4** — data bugs are insidious because they look like model underfitting
5. **Categories 5 & 6** — inference and test quality last

---

## Status

- [x] Category 1 — Diffusion Math (clean — no bugs)
- [x] Category 2 — STAR Model Architecture (1 bug fixed: frame_idx RoPE mismatch)
- [x] Category 3 — Loss & Training Loop (1 bug fixed: LoRA O(d²) forward)
- [ ] Category 4 — Data Pipeline
- [ ] Category 5 — Inference
- [ ] Category 6 — Tests
