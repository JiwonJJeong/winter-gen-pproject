# Notebook Usage Guide

## Available Notebooks

### 1. colab_single_protein_ddpm.ipynb ⭐ (Recommended)

**Purpose:** Train DDPM on a single protein with git clone workflow

**Features:**
- ✅ Git clone as primary method
- ✅ Single-protein focus (specify protein in config)
- ✅ Dynamic data creation (no data upload needed)
- ✅ Complete workflow from setup to results
- ✅ Hyperparameter configuration upfront

**Usage:**
1. Push your code to GitHub
2. Open notebook in Colab from GitHub
3. Change `REPO_URL` in Step 2
4. Configure protein in Step 3
5. Run all cells

**Best for:** Production use, collaboration, reproducibility

---

### 2. test_dataset_loader.ipynb

**Purpose:** Test and debug the dataset loader

**Best for:** Development and debugging

---

## Quick Start (3 Steps)

### Step 1: Push to GitHub

```bash
# In your project directory
git add gen_model/ colab_single_protein_ddpm.ipynb
git commit -m "Add DDPM training code"
git push origin main
```

**Important:** Do NOT push `data/` - it's created dynamically!

### Step 2: Open in Colab

**Method A: Direct Link** (Recommended)
```
https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_single_protein_ddpm.ipynb
```

**Method B: From Colab UI**
1. Go to https://colab.research.google.com
2. File → Open notebook → GitHub
3. Enter your repo URL
4. Select `colab_single_protein_ddpm.ipynb`

### Step 3: Configure and Run

1. **Enable GPU:** Runtime → Change runtime type → GPU
2. **Update Step 2:** Change `REPO_URL` to your repo
3. **Configure Step 3:** Set protein name and parameters
4. **Run all:** Runtime → Run all (or Ctrl+F9)

## Notebook Structure

```
Step 1: Environment Setup
├── Install dependencies
└── Check GPU

Step 2: Clone Repository ⭐
├── Git clone from GitHub (primary)
└── Alternative: Upload zip (optional)

Step 3: Protein Configuration
├── Protein name
├── Number of frames/residues
├── Hyperparameters
└── Training settings

Step 4: Create/Load Data
├── Option A: Download real data
└── Option B: Create synthetic data (default)

Steps 5-8: Training
├── Load dataset
├── Create model
├── Train DDPM
└── Save checkpoints

Steps 9-12: Evaluation & Results
├── Test denoising
├── Generate samples
├── Visualize results
└── Download outputs
```

## Configuration

### Protein Settings (Step 3)

```python
'protein': {
    'name': '4o66_C',      # Your protein name
    'replica': 1,           # Replica number
    'num_frames': 200,      # Trajectory frames
    'num_residues': 100,    # Protein size
}
```

### Hyperparameters (Step 3)

| Parameter | Quick Test | Balanced ⭐ | High Quality |
|-----------|-----------|-----------|--------------|
| timesteps | 100 | 500 | 1000 |
| hidden_dim | 128 | 256 | 512 |
| batch_size | 16 | 8 | 4 |
| num_epochs | 50 | 100 | 200 |
| **Time (T4)** | 30-60 min | 2-4 hours | 8-12 hours |

## Git Clone Details

### Why Git Clone?

✅ **Advantages:**
- Version control built-in
- Easy to update (`git pull`)
- No file size limits
- Can push changes back
- Better collaboration
- No re-upload on disconnect

❌ **Zip Upload Disadvantages:**
- Need to re-upload if session disconnects
- Manual version tracking
- File size limits
- No change history

### Repository URL Format

**Public Repository:**
```python
REPO_URL = "https://github.com/username/repo.git"
```

**Private Repository (with token):**
```python
# Generate token at: https://github.com/settings/tokens
# Store in Colab secrets (🔑 icon)
from google.colab import userdata
TOKEN = userdata.get('GITHUB_TOKEN')
REPO_URL = f"https://{TOKEN}@github.com/username/repo.git"
```

### Common Repo Structures

**Structure 1: gen_model in root** (No navigation needed)
```
your-repo/
├── gen_model/
├── colab_single_protein_ddpm.ipynb
└── README.md
```

**Structure 2: gen_model in subdirectory** (Need navigation)
```
your-repo/
└── project/
    ├── gen_model/
    └── colab_single_protein_ddpm.ipynb
```

For Structure 2, uncomment in Step 2:
```python
%cd project  # Navigate to subdirectory
```

### Colab Resource Limits

**Free Tier:**
- **GPU**: NVIDIA T4 (16GB)
- **RAM**: ~12GB
- **Disk**: ~100GB
- **Session**: 12 hours max
- **Recommended config**: Quick Test or Balanced

**Colab Pro ($10/month):**
- **GPU**: V100 (32GB) or better
- **RAM**: ~25GB
- **Disk**: ~200GB
- **Session**: 24 hours max
- **Recommended config**: Balanced or High Quality

**Colab Pro+ ($50/month):**
- **GPU**: A100 (40GB) or V100
- **RAM**: ~50GB
- **Disk**: ~200GB
- **Session**: No hard limit
- **Recommended config**: Any, including High Quality

### Dealing with Disconnections

Colab can disconnect after inactivity or when session limit is reached.

**Auto-Reconnect Script:**

Add this cell at the beginning of your notebook:

```javascript
%%javascript
function ClickConnect(){
  console.log("Working");
  document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect,60000)
```

This clicks the connect button every 60 seconds to prevent disconnection.

**Save Checkpoints Frequently:**

The notebook saves checkpoints every 10 epochs by default. To save more frequently, change in Step 3:

```python
'training': {
    'save_every': 5,  # Save every 5 epochs instead of 10
}
```

**Resume Training After Disconnect:**

If training is interrupted:

1. Re-run cells 1-6 (setup and config)
2. Modify the checkpoint loading cell:
```python
# Find the latest checkpoint
import glob
checkpoint_files = sorted(glob.glob("checkpoints/simple_ddpm/*.pt"))
if checkpoint_files:
    # Load the latest checkpoint and continue training
    latest_checkpoint = checkpoint_files[-1]
    print(f"Resuming from: {latest_checkpoint}")
```
3. Continue training from that checkpoint

### Pro Tips

**Tip 1: Use a Dev Branch**

```bash
# Create a dev branch for experiments
git checkout -b colab-dev
git push origin colab-dev

# In Colab, clone specific branch:
!git clone -b colab-dev https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

**Tip 2: Auto-Pull Latest Code**

Add this to Step 2 to automatically pull updates:

```python
import os
if os.path.exists('gen_model'):
    !cd YOUR_REPO && git pull
else:
    !git clone {REPO_URL}
```

**Tip 3: Save Results Back to GitHub**

After training, commit results:

```python
# In Colab
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"

!git add checkpoints/ outputs/
!git commit -m "Training results for {protein_name}"
!git push  # Use personal access token for authentication
```

**Tip 4: Save to Google Drive**

For persistent storage across sessions:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Google Drive
!cp -r checkpoints /content/drive/MyDrive/ddpm_checkpoints
!cp -r outputs /content/drive/MyDrive/ddpm_outputs

print("✓ Results saved to Google Drive")
```

## Data Handling

### Option A: Use Synthetic Data (Default)

```python
'use_real_data': False  # Creates dummy data automatically
```

**Use for:**
- Testing the pipeline
- Code development
- Hyperparameter tuning
- CI/CD testing

### Option B: Use Real Data

```python
'use_real_data': True
'data_source': 'YOUR_URL_OR_PATH'
```

Then customize Step 4's download section:

**From URL:**
```python
!wget -O data.tar.gz {protein_config.data_source}
!tar -xzf data.tar.gz
```

**From Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/md_data/{prot_cfg.name} data/
```

**From Google Drive File ID:**
```python
!pip install -q gdown
!gdown {protein_config.data_source} -O data.zip
!unzip -q data.zip
```

## Updates and Changes

### Updating Code During Session

If you make changes to your GitHub repo:

```python
# Option 1: Pull latest changes
%cd YOUR_REPO
!git pull
%cd /content

# Option 2: Restart and re-clone
# Runtime → Restart runtime
# Then re-run all cells
```

### Pushing Results Back

To save training results to GitHub:

```python
# Configure git
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"

# Commit and push
!git add outputs/ checkpoints/
!git commit -m "Training results for {protein_name}"
!git push  # Use token for authentication
```

## Troubleshooting

### Issue: "Repository not found"

**Check:**
1. Repository URL is correct
2. Repository is public (or you have a token for private)
3. Token has correct permissions

**Fix:**
```python
# Test the URL
!git ls-remote {REPO_URL}
```

### Issue: "gen_model not found"

**Check:**
1. `gen_model/` exists in your repo
2. Repo structure matches expectation
3. Git clone succeeded without errors

**Fix:**
```python
# Verify structure
!ls -la
!find . -name "simple_train.py"
```

### Issue: "Module not found"

**Check:**
1. All dependencies installed (Step 1)
2. Python path includes current directory

**Fix:**
```python
import sys
sys.path.insert(0, '.')  # Or specific path
```

### Issue: Training is slow

**Check:**
1. GPU is enabled (Runtime → Change runtime type)
2. Verify GPU is being used:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

**Fix:**
- Reduce `batch_size` if OOM
- Reduce `hidden_dim` for faster training
- Reduce `timesteps` for testing

## Best Practices

1. **Always use GPU** - Enable before running
2. **Test first** - Run 5-10 epochs before full training
3. **Save frequently** - Set `save_every=5` for long runs
4. **Use git** - Better than uploading zips
5. **Version control** - Commit hyperparameter changes
6. **Document experiments** - Note configs in git commit messages
7. **Monitor progress** - Check loss curves and samples
8. **Download results** - Don't rely on session staying alive

## Example Workflow

```python
# ========== LOCAL ==========
# 1. Push code
git add gen_model/ colab_single_protein_ddpm.ipynb
git commit -m "Add DDPM training"
git push

# ========== COLAB ==========
# 2. Open notebook from GitHub
# 3. Enable GPU
# 4. Update REPO_URL in Step 2
REPO_URL = "https://github.com/username/repo.git"

# 5. Configure protein in Step 3
'protein': {'name': 'my_protein', ...}

# 6. Run all cells
# Runtime → Run all

# 7. Wait for training (~2-4 hours)
# 8. Download results (automatic at end)

# ========== LOCAL ==========
# 9. Extract results.zip
# 10. Analyze outputs
```

## Resources

- [Google Colab](https://colab.research.google.com/)
- [GitHub Docs](https://docs.github.com/)
- [Colab Secrets](https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75)
- [HYPERPARAMETER_GUIDE.md](HYPERPARAMETER_GUIDE.md) - Hyperparameter tuning guide
- [gen_model/README.md](gen_model/README.md) - Code documentation
