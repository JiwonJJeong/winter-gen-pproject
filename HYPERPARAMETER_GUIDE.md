# Hyperparameter Tuning Guide for DDPM Training

This guide helps you understand and tune the hyperparameters in the simple DDPM training setup.

## Quick Reference Table

| Category | Parameter | Default | Range | Impact on Training | Impact on Quality |
|----------|-----------|---------|-------|-------------------|-------------------|
| **Diffusion** | timesteps | 1000 | 100-1000 | Higher = slower inference | More steps = smoother denoising |
| | beta_start | 0.0001 | 0.0001-0.001 | Minimal | Lower = gradual initial noising |
| | beta_end | 0.02 | 0.01-0.05 | Minimal | Higher = more final noise |
| **Model** | hidden_dim | 256 | 128-512 | Larger = slower, more memory | Larger = more capacity |
| | time_emb_dim | 128 | 64-256 | Minimal impact | Larger = better time conditioning |
| **Training** | batch_size | 8 | 2-32 | Larger = faster but needs GPU memory | Larger = more stable gradients |
| | num_epochs | 100 | 50-200 | More = longer training | More = better but can overfit |
| | learning_rate | 1e-4 | 1e-5 to 1e-3 | Too high = unstable | Needs tuning for stability |
| | num_workers | 2 | 0-4 | More = faster data loading | No impact |

## Detailed Parameter Descriptions

### 1. Diffusion Parameters

#### `timesteps` (Default: 1000)
**What it does:** Number of steps in the forward diffusion process (adding noise) and reverse process (denoising).

**How to tune:**
- **100 steps**: Fast training and inference, lower quality
  - Use for: Quick prototyping, debugging
- **500 steps**: Balanced speed/quality
  - Use for: Most experiments, good baseline
- **1000 steps**: Slow but high quality
  - Use for: Final models, publication-quality results

**Trade-offs:**
- ⬆️ More steps = Better sample quality, slower inference
- ⬇️ Fewer steps = Faster inference, potentially lower quality

#### `beta_start` (Default: 0.0001)
**What it does:** Controls the noise level at the first diffusion step (t=0).

**How to tune:**
- **0.0001**: Very gradual noise addition (recommended)
- **0.001**: Faster noise addition
- **0.0005**: Middle ground

**Trade-offs:**
- ⬆️ Higher = Faster convergence but may skip details
- ⬇️ Lower = More gradual, better captures fine details

#### `beta_end` (Default: 0.02)
**What it does:** Controls the noise level at the final diffusion step (t=T).

**How to tune:**
- **0.01**: Conservative noise level
- **0.02**: Standard (recommended)
- **0.05**: Aggressive noise level

**Trade-offs:**
- ⬆️ Higher = More noise at end, potentially harder to learn
- ⬇️ Lower = Less noise, may not fully destroy signal

**Alternative:** Consider implementing cosine noise schedule for potentially better results.

### 2. Model Architecture Parameters

#### `hidden_dim` (Default: 256)
**What it does:** Size of hidden layers in the encoder-decoder network.

**How to tune:**
- **128**: Small, fast, low memory
  - Model params: ~500K
  - Use for: Quick experiments, limited GPU memory
- **256**: Medium (recommended)
  - Model params: ~2M
  - Use for: Most cases, good balance
- **512**: Large, high capacity
  - Model params: ~8M
  - Use for: Complex data, large datasets

**Trade-offs:**
- ⬆️ Larger = More capacity, can learn complex patterns, slower, more memory
- ⬇️ Smaller = Faster, less memory, may underfit on complex data

**GPU Memory Impact:**
```
hidden_dim=128: ~2-4 GB GPU memory (batch_size=16)
hidden_dim=256: ~4-8 GB GPU memory (batch_size=8)
hidden_dim=512: ~8-16 GB GPU memory (batch_size=4)
```

#### `time_emb_dim` (Default: 128)
**What it does:** Dimension of the timestep embedding vector.

**How to tune:**
- **64**: Minimal embedding
- **128**: Standard (recommended)
- **256**: Large embedding

**Trade-offs:**
- ⬆️ Larger = Better time conditioning, minimal overhead
- ⬇️ Smaller = Faster, but may not capture temporal information well

**General rule:** Set to `hidden_dim // 2` or `hidden_dim`

### 3. Training Parameters

#### `batch_size` (Default: 8)
**What it does:** Number of samples processed in parallel per training iteration.

**How to tune based on GPU:**
- **Colab Free (T4, 15GB)**: batch_size=4-8
- **Colab Pro (V100, 32GB)**: batch_size=16-32
- **Colab Pro+ (A100, 40GB)**: batch_size=32-64
- **CPU only**: batch_size=1-2

**How to find max batch size:**
1. Start with batch_size=2
2. Double until you get OOM (Out of Memory)
3. Use 50-75% of that value

**Trade-offs:**
- ⬆️ Larger = Faster training, more stable gradients, needs more memory
- ⬇️ Smaller = Slower training, less stable, works with limited memory

#### `num_epochs` (Default: 100)
**What it does:** Number of complete passes through the training dataset.

**How to tune:**
- **50 epochs**: Quick baseline
  - Use for: Initial experiments, debugging
- **100 epochs**: Standard training
  - Use for: Most experiments
- **200+ epochs**: Extended training
  - Use for: Final models, may need early stopping

**How to know when to stop:**
- Monitor validation loss (if implemented)
- Look at generated samples quality
- Watch for signs of overfitting

**Trade-offs:**
- ⬆️ More = Better training, risk of overfitting, longer time
- ⬇️ Fewer = Faster, may underfit

#### `learning_rate` (Default: 1e-4)
**What it does:** Step size for parameter updates during optimization.

**How to tune:**
- **1e-5**: Very conservative, slow learning
  - Use for: Fine-tuning, unstable training
- **1e-4**: Standard (recommended)
  - Use for: Most cases
- **1e-3**: Aggressive, fast learning
  - Use for: Quick experiments, may be unstable

**How to find optimal learning rate:**
1. Start with 1e-4
2. If loss oscillates → decrease by 10x
3. If loss decreases very slowly → increase by 2-5x
4. Use learning rate warmup for large models

**Trade-offs:**
- ⬆️ Higher = Faster convergence, risk of instability/divergence
- ⬇️ Lower = More stable, slower convergence

**Advanced:** Implement learning rate scheduling:
```python
# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)

# Step decay
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)
```

#### `num_workers` (Default: 2)
**What it does:** Number of parallel processes for data loading.

**How to tune:**
- **0**: Single-threaded (good for debugging)
- **2**: Standard for Colab
- **4**: Maximum for most systems
- **Higher**: May not help or cause issues

**Trade-offs:**
- ⬆️ More = Faster data loading, more CPU/memory
- ⬇️ Fewer = Slower data loading, less overhead

**Note:** On Colab, 2-4 is usually optimal. Higher values don't help much.

## Recommended Configurations

### Configuration 1: Quick Test (Debug/Prototype)
```python
config = {
    'diffusion': {
        'timesteps': 100,
        'beta_start': 0.0001,
        'beta_end': 0.02,
    },
    'model': {
        'hidden_dim': 128,
        'time_emb_dim': 64,
    },
    'training': {
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'num_workers': 2,
    },
}
```
**Training time:** ~30-60 min on Colab T4
**Use case:** Quick prototyping, testing code changes

### Configuration 2: Balanced (Recommended Default)
```python
config = {
    'diffusion': {
        'timesteps': 500,
        'beta_start': 0.0001,
        'beta_end': 0.02,
    },
    'model': {
        'hidden_dim': 256,
        'time_emb_dim': 128,
    },
    'training': {
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'num_workers': 2,
    },
}
```
**Training time:** ~2-4 hours on Colab T4
**Use case:** Most experiments, good quality

### Configuration 3: High Quality (Final Model)
```python
config = {
    'diffusion': {
        'timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
    },
    'model': {
        'hidden_dim': 512,
        'time_emb_dim': 256,
    },
    'training': {
        'batch_size': 4,
        'num_epochs': 200,
        'learning_rate': 5e-5,
        'num_workers': 2,
    },
}
```
**Training time:** ~8-12 hours on Colab V100
**Use case:** Publication-quality results, final models

### Configuration 4: Memory-Constrained (Small GPU)
```python
config = {
    'diffusion': {
        'timesteps': 200,
        'beta_start': 0.0001,
        'beta_end': 0.02,
    },
    'model': {
        'hidden_dim': 128,
        'time_emb_dim': 64,
    },
    'training': {
        'batch_size': 4,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'num_workers': 0,
    },
}
```
**Training time:** ~2-3 hours on Colab T4
**Use case:** Limited GPU memory, CPU-only training

## Troubleshooting Common Issues

### Issue 1: Out of Memory (OOM)
**Symptoms:** CUDA out of memory error during training

**Solutions (try in order):**
1. Reduce `batch_size` by half (8 → 4 → 2)
2. Reduce `hidden_dim` (512 → 256 → 128)
3. Set `num_workers=0`
4. Clear GPU cache: `torch.cuda.empty_cache()`

### Issue 2: Training Too Slow
**Symptoms:** Each epoch takes >30 minutes

**Solutions:**
1. Increase `batch_size` (if GPU memory allows)
2. Reduce `timesteps` (1000 → 500 → 200)
3. Reduce `hidden_dim` if quality is acceptable
4. Increase `num_workers` (2 → 4)
5. Use mixed precision training (add to code)

### Issue 3: Loss Not Decreasing
**Symptoms:** Loss stays constant or increases

**Solutions:**
1. Reduce `learning_rate` by 10x
2. Check data loading (verify dataset is correct)
3. Increase `hidden_dim` (model may be too small)
4. Try different initialization
5. Add gradient clipping

### Issue 4: Poor Sample Quality
**Symptoms:** Generated samples are noisy or unrealistic

**Solutions:**
1. Increase `num_epochs` (train longer)
2. Increase `timesteps` (500 → 1000)
3. Increase `hidden_dim` (more capacity)
4. Decrease `learning_rate` (more stable training)
5. Try different `beta_start` and `beta_end` values

### Issue 5: Overfitting
**Symptoms:** Training loss decreases but validation loss increases

**Solutions:**
1. Reduce `num_epochs`
2. Reduce `hidden_dim` (model too large)
3. Add dropout or weight decay (requires code modification)
4. Increase dataset size
5. Implement early stopping

## Advanced Tuning Strategies

### 1. Learning Rate Finder
Run a quick experiment to find optimal learning rate:

```python
lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
for lr in lrs:
    # Train for 10 epochs
    # Record final loss
    # Choose LR with best loss
```

### 2. Progressive Training
Start with easy configuration, gradually increase difficulty:

```python
# Phase 1: Quick training with low timesteps
train(timesteps=100, epochs=50)

# Phase 2: Increase timesteps
train(timesteps=500, epochs=50)

# Phase 3: Full quality
train(timesteps=1000, epochs=100)
```

### 3. Hyperparameter Search
Use grid search or random search:

```python
# Grid search example
for hidden_dim in [128, 256, 512]:
    for lr in [1e-5, 1e-4, 1e-3]:
        train(hidden_dim=hidden_dim, lr=lr)
```

### 4. Model Scaling Rules
When scaling model size:

```
If you double hidden_dim:
- Expect 4x more parameters
- Need 2x more GPU memory
- Can reduce batch_size by 2x
- May need 0.5x learning_rate
```

## Monitoring Training

### Key Metrics to Watch

1. **Training Loss**
   - Should decrease steadily
   - Typical range: 0.01 - 0.1 after convergence

2. **Sample Quality**
   - Generate samples every N epochs
   - Visual inspection of reconstructions

3. **MSE (Mean Squared Error)**
   - Reconstruction error on validation set
   - Lower is better, typical range: 0.001 - 0.01

4. **Training Time per Epoch**
   - Should be consistent
   - Sudden increases = potential issue

## Summary: Where to Start

**If you're new:**
1. Start with **Balanced Configuration** (Config 2)
2. Train for 50 epochs first
3. Evaluate sample quality
4. Adjust based on results

**If training is too slow:**
1. Use **Quick Test Configuration** (Config 1)
2. Reduce `num_epochs` to 30-50
3. Accept lower quality for faster iteration

**If you need best quality:**
1. Use **High Quality Configuration** (Config 3)
2. Train for 150-200 epochs
3. May need Colab Pro for better GPU

**If you have limited resources:**
1. Use **Memory-Constrained Configuration** (Config 4)
2. Consider training in multiple sessions
3. Save checkpoints frequently

## Resources

- [DDPM Paper](https://arxiv.org/abs/2006.11239) - Original DDPM paper
- [Improved DDPM](https://arxiv.org/abs/2102.09672) - Improvements to DDPM
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html) - PyTorch documentation
- [Hyperparameter Tuning](https://cs231n.github.io/neural-networks-3/) - Stanford CS231n guide
