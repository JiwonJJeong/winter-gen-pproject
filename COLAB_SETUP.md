# Google Colab Setup Guide

Quick guide to get your DDPM training running on Google Colab.

## Step-by-Step Setup

### 1. Prepare Your Data Locally

First, create a zip file with the necessary directories:

```bash
# In your project directory
zip -r gen_model.zip gen_model/ data/
```

This will create `gen_model.zip` containing:
- `gen_model/` - All training and inference code
- `data/` - Your MD trajectory data

### 2. Open the Notebook in Colab

**Option A: Upload the notebook**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `colab_ddpm_training.ipynb`

**Option B: From GitHub (if you have a repo)**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Open notebook**
3. Click **GitHub** tab
4. Enter your repository URL
5. Select the notebook

### 3. Enable GPU

⚠️ **Important:** Make sure GPU is enabled!

1. In Colab, go to **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4 or better)
3. Click **Save**

### 4. Run the Notebook

Simply run all cells in order:

1. **Cell 1-2**: Install dependencies and check GPU
2. **Cell 3**: Upload `gen_model.zip` when prompted
3. **Cell 4-6**: Configure hyperparameters (adjust as needed)
4. **Cell 7-10**: Run training
5. **Cell 11-14**: Run inference and evaluation
6. **Cell 15**: Download results

## File Upload Options

### Option A: Upload Zip File (Recommended for Colab)

**Pros:** Simple, everything in one file
**Cons:** Need to re-upload if session disconnects

**Steps:**
1. Create zip: `zip -r gen_model.zip gen_model/ data/`
2. Run upload cell in notebook
3. Select `gen_model.zip` from your computer

### Option B: Mount Google Drive

**Pros:** Persistent storage, no re-upload needed
**Cons:** Requires Google Drive setup

**Steps:**
1. Upload `gen_model.zip` to your Google Drive
2. Add this cell to notebook:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/gen_model.zip .
!unzip -q gen_model.zip
```

### Option C: Clone from GitHub

**Pros:** Easy version control, can push changes
**Cons:** Need GitHub repo

**Steps:**
1. Push your code to GitHub
2. In notebook, replace the git clone cell with your repo URL

## Colab Resource Limits

### Free Tier
- **GPU**: NVIDIA T4 (16GB)
- **RAM**: ~12GB
- **Disk**: ~100GB
- **Session**: 12 hours max
- **Recommended config**: Quick Test or Balanced

### Colab Pro ($10/month)
- **GPU**: V100 (32GB) or better
- **RAM**: ~25GB
- **Disk**: ~200GB
- **Session**: 24 hours max
- **Recommended config**: Balanced or High Quality

### Colab Pro+ ($50/month)
- **GPU**: A100 (40GB) or V100
- **RAM**: ~50GB
- **Disk**: ~200GB
- **Session**: No hard limit
- **Recommended config**: Any, including High Quality

## Expected Training Times

Based on default dataset size (~1000 frames):

| Configuration | Colab Free (T4) | Colab Pro (V100) |
|---------------|-----------------|------------------|
| Quick Test (50 epochs) | ~30-60 min | ~15-30 min |
| Balanced (100 epochs) | ~2-4 hours | ~1-2 hours |
| High Quality (200 epochs) | ~8-12 hours | ~4-6 hours |

**Note:** Times vary based on dataset size and complexity.

## Dealing with Disconnections

Colab can disconnect after inactivity or when session limit is reached.

### Auto-Reconnect Script

Add this cell at the beginning of your notebook:

```javascript
%%javascript
function ClickConnect(){
  console.log("Working");
  document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect,60000)
```

This clicks the connect button every 60 seconds.

### Save Checkpoints Frequently

The notebook saves checkpoints every 10 epochs by default. To save more frequently:

```python
config.training.save_every = 5  # Save every 5 epochs instead of 10
```

### Resume Training After Disconnect

If training is interrupted:

1. Re-run cells 1-6 (setup and config)
2. Modify the checkpoint loading cell:
```python
# Find the latest checkpoint
checkpoint_files = sorted(glob.glob("checkpoints/simple_ddpm/*.pt"))
if checkpoint_files:
    config.checkpoint.load_from = checkpoint_files[-1]
    print(f"Resuming from: {config.checkpoint.load_from}")
```
3. Continue training from that checkpoint

## Hyperparameter Tuning on Colab

### Quick Iteration Strategy

1. **Start small** (Quick Test config):
   ```python
   config.diffusion.timesteps = 100
   config.model.hidden_dim = 128
   config.training.num_epochs = 30
   ```

2. **Test hyperparameter**:
   - Change ONE parameter at a time
   - Train for 20-30 epochs
   - Evaluate quality

3. **Scale up**:
   - Once you find good settings, increase epochs
   - Use Balanced or High Quality config

### Example: Testing Learning Rates

```python
# Test different learning rates quickly
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4]

for lr in learning_rates:
    print(f"\n{'='*80}")
    print(f"Testing learning_rate={lr}")
    print('='*80)

    config.training.learning_rate = lr
    config.training.num_epochs = 20  # Quick test
    config.checkpoint.save_dir = f'checkpoints/lr_{lr}'

    # Run training...
    # Compare results
```

## Downloading Results

### Option 1: Direct Download (Automatic)

The notebook automatically downloads results as `results.zip` at the end.

### Option 2: Manual Download

Right-click on files in the file browser (left sidebar) and select **Download**.

### Option 3: Save to Google Drive

Add this cell after training:

```python
# Copy results to Google Drive
from google.colab import drive
drive.mount('/content/drive')

!cp -r checkpoints /content/drive/MyDrive/ddpm_checkpoints
!cp -r outputs /content/drive/MyDrive/ddpm_outputs

print("✓ Results saved to Google Drive")
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
1. Reduce `batch_size` (8 → 4 → 2)
2. Reduce `hidden_dim` (512 → 256 → 128)
3. Restart runtime: **Runtime → Restart runtime**

### Issue: "Session disconnected"
**Solution:**
1. Check internet connection
2. Use auto-reconnect script (see above)
3. Save to Google Drive periodically

### Issue: "No such file or directory: gen_model/"
**Solution:**
1. Make sure you uploaded `gen_model.zip`
2. Check that unzip worked: `!ls -la`
3. Verify files: `!ls gen_model/`

### Issue: "Cannot import module"
**Solution:**
1. Re-run the imports cell
2. Check Python path: `import sys; print(sys.path)`
3. Verify files exist: `!ls gen_model/*.py`

### Issue: Training is very slow
**Solution:**
1. Check GPU is enabled (Runtime → Change runtime type)
2. Verify GPU is being used:
   ```python
   print(torch.cuda.is_available())
   print(torch.cuda.current_device())
   ```
3. Reduce `num_workers` to 0

## Best Practices

1. **Save frequently**: Set `save_every=5` instead of 10
2. **Use Google Drive**: For long training runs
3. **Monitor GPU**: Check GPU usage in Runtime → Manage sessions
4. **Test first**: Run 5-10 epochs before full training
5. **Download results**: Don't rely on session staying alive
6. **Version control**: Push code to GitHub when making changes

## Example Workflow

Here's a complete workflow for running experiments:

```python
# 1. Quick test (30 min)
config.training.num_epochs = 30
config.diffusion.timesteps = 100
train()  # Check if code works

# 2. Tune hyperparameters (2 hours)
config.training.num_epochs = 50
for hidden_dim in [128, 256]:
    config.model.hidden_dim = hidden_dim
    train()
    evaluate()

# 3. Full training (4 hours)
config.training.num_epochs = 100
config.diffusion.timesteps = 500
config.model.hidden_dim = 256  # Best from step 2
train()

# 4. Final evaluation
evaluate_on_test_set()
download_results()
```

## Getting Help

If you encounter issues:

1. Check error messages carefully
2. Re-run cells in order
3. Restart runtime if needed
4. Verify GPU is enabled
5. Check file paths are correct

## Additional Resources

- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Colab Best Practices](https://github.com/googlecolab/colabtools/wiki)
- [PyTorch Colab Tutorial](https://pytorch.org/tutorials/beginner/colab.html)
- [Hyperparameter Guide](HYPERPARAMETER_GUIDE.md) (in this repo)
