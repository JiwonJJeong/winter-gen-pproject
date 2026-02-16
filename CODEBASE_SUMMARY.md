# Codebase Structure Summary

This document provides a comprehensive overview of the `model/`, `data_se3/`, and `experiments/` directories, explaining each file's purpose and how they interact to implement SE(3) diffusion for protein structure generation.

---

## Overview

This codebase implements **SE(3) diffusion models** for protein structure generation, adapted from the [jasonkyuyim/se3_diffusion](https://github.com/jasonkyuyim/se3_diffusion) repository. The system uses diffusion processes on the SE(3) group (3D rotations and translations) to generate protein backbone structures.

### Key Concepts
- **SE(3)**: Special Euclidean group in 3D - combines rotations (SO(3)) and translations (R³)
- **Diffusion Models**: Generative models that learn to reverse a noise-adding process
- **IPA (Invariant Point Attention)**: Attention mechanism that respects 3D geometric structure

---

## Directory Structure

```
model/              # Neural network architectures
data_se3/data/      # Data processing and diffusion mathematics
experiments/        # Training and inference scripts
```

---

## 1. `model/` Directory

Contains the neural network architectures for the score prediction model.

### `ipa_pytorch.py` (674 lines)
**Purpose**: Implements Invariant Point Attention and the main score network architecture.

**Key Components**:
- **`InvariantPointAttention`**: Core attention mechanism that operates on 3D structures
  - Takes single representation `s`, pair representation `z`, and rigid transformations `r`
  - Computes attention in both scalar space and 3D point space
  - Outputs updated single representation
  
  **Understanding the Input Representations**:
  
  1. **Single Representation `s`** - Shape: `[batch, N_residues, C_s]`
     - Per-residue feature vector (like a node embedding in a graph)
     - Contains information about each individual residue
     - Examples: amino acid type, local structure features, learned embeddings
     - Analogous to "node features" in graph neural networks
  
  2. **Pair Representation `z`** - Shape: `[batch, N_residues, N_residues, C_z]`
     - Pairwise features between all residue pairs (like edge features in a graph)
     - Contains information about relationships between residues i and j
     - Examples: sequence separation, distance bins, contact predictions
     - Analogous to "edge features" in graph neural networks
     - Symmetric or asymmetric depending on the features
  
  3. **Rigid Transformations `r`** - Shape: `[batch, N_residues]` (each is a Rigid object)
     - 3D coordinate frame for each residue (rotation + translation)
     - Represents the local geometry: where the residue is and how it's oriented
     - Each rigid transformation has 6 degrees of freedom:
       - 3 for rotation (SO(3) - can be represented as rotation matrix or quaternion)
       - 3 for translation (R³ - x, y, z position)
     - Typically defined by the backbone atoms (N, CA, C)
     - Allows the model to reason about 3D geometric relationships
  
  **Why IPA needs all three**:
  - `s` captures local residue information
  - `z` captures pairwise relationships and constraints
  - `r` provides the 3D geometric context
  - IPA combines all three to perform geometry-aware attention
  
- **`IpaScore`**: Main model that predicts diffusion scores
  - Stacks multiple IPA blocks with transformer layers
  - Predicts rotation scores and translation scores
  - Includes backbone update mechanism
  
- **`BackboneUpdate`**: Predicts rigid body updates (6 DOF: 3 rotation + 3 translation)

- **`TorsionAngles`**: Predicts psi torsion angles for side chains

- **Helper Layers**:
  - `Linear`: Custom linear layer with various initialization schemes
  - `StructureModuleTransition`: MLP with residual connections
  - `EdgeTransition`: Updates edge (pair) representations

**Interactions**:
- Used by `score_network.py` to build the complete model
- Receives embeddings from `score_network.Embedder`
- Uses `all_atom.py` for geometric calculations

---

### `score_network.py` (217 lines)
**Purpose**: Wraps the IPA architecture with input embedding and output processing.

**Key Components**:
- **`Embedder`**: Converts raw inputs into node and edge embeddings
  - Embeds timestep `t` using sinusoidal encoding (similar to Transformer positional encoding)
  - Embeds sequence positions (residue indices along the chain)
  - **Self-conditioning** (optional): Embeds previous model predictions to improve sampling
    - During training: With some probability, run the model twice - first to get a prediction, then again using that prediction as an additional input
    - During inference: Use the prediction from the previous denoising step as input to the current step
    - Implemented as a distance histogram (distogram) between predicted Cα atoms
    - Helps the model make more consistent predictions across denoising steps
    - Inspired by "Analog Bits" self-conditioning technique
  - Outputs: 
    - `node_embed` [B, N, D_node]: Per-residue features ready for IPA (becomes the single representation `s`)
    - `edge_embed` [B, N, N, D_edge]: Pairwise features ready for IPA (becomes the pair representation `z`)

- **`ScoreNetwork`**: Main model class
  - Combines `Embedder` + `IpaScore`
  - Takes noised protein structures and predicts denoising scores
  - Outputs: rotation scores, translation scores, predicted structures

**Interactions**:
- Instantiated in `train_se3_diffusion.py` for training
- Uses `ipa_pytorch.IpaScore` as the core model
- Receives diffuser from `se3_diffuser.py`
- Uses `all_atom.py` for computing backbone atom positions

---

### `layers.py` (364 lines)
**Purpose**: Provides reusable neural network building blocks.

**Key Components**:
- **`Dense`**: Linear layer with activation and custom initialization
- **`ScaledSiLU`**: SiLU activation scaled by 1/0.6
- **`ResidualLayer`**: Residual MLP block scaled by 1/√2
- **`EfficientInteractionDownProjection`**: Projects spherical harmonics for geometric features
- Weight initialization functions: `he_orthogonal_init`, etc.

**Interactions**:
- Utility layers that could be used throughout the model
- Contains duplicate functions from `ipa_pytorch.py` (permute_final_dims, etc.)

---

## 2. `data_se3/data/` Directory

Contains all data processing, diffusion mathematics, and dataset loading code.

### Diffusion Mathematics

#### `se3_diffuser.py` (270 lines)
**Purpose**: Implements SE(3) diffusion - combines rotation and translation diffusion.

**Key Components**:
- **`SE3Diffuser`**: Main class coordinating SO(3) and R³ diffusion
  - `forward_marginal()`: Adds noise to clean structures → noised structures at time t
  - `reverse()`: Denoising step from time t to t-1
  - `calc_rot_score()`: Computes rotation score (gradient of log density)
  - `calc_trans_score()`: Computes translation score
  - `sample_ref()`: Samples from reference distribution (pure noise)

**Interactions**:
- Composes `so3_diffuser.SO3Diffuser` and `r3_diffuser.R3Diffuser`
- Used by training/inference scripts to noise and denoise structures
- Used by `score_network.py` to compute ground truth scores

---

#### `so3_diffuser.py` (368 lines)
**Purpose**: Implements diffusion on SO(3) (rotation group).

**Key Components**:
- **`SO3Diffuser`**: Rotation diffusion using IGSO(3) distribution
  - `sigma(t)`: Variance schedule (linear or logarithmic)
  - `sample()`: Samples rotation noise as axis-angle vectors
  - `score()`: Computes score of IGSO(3) density
  - `forward_marginal()`: Adds rotation noise
  - `reverse()`: Geodesic random walk for denoising

- **`igso3_expansion()`**: Truncated power series for IGSO(3) density
- **`score()`**: Derivative of log density using quotient rule

**Interactions**:
- Used by `SE3Diffuser` for rotation component
- Relies on `igso3.py` for precomputed IGSO(3) quantities
- Uses `so3_utils.py` for SO(3) operations

---

#### `r3_diffuser.py` (168 lines)
**Purpose**: Implements diffusion on R³ (translation space).

**Key Components**:
- **`R3Diffuser`**: VP-SDE (Variance Preserving SDE) for translations
  - `b_t(t)`: Linear variance schedule
  - `forward_marginal()`: Adds Gaussian noise to positions
  - `reverse()`: Reverse SDE step with Langevin dynamics
  - `score()`: Computes translation score
  - `calc_trans_0()`: Predicts clean positions from noised positions and score

**Interactions**:
- Used by `SE3Diffuser` for translation component
- Scales coordinates using `coordinate_scaling` parameter

---

#### `igso3.py` (219 lines)
**Purpose**: Precomputes and caches IGSO(3) distribution quantities.

**Key Components**:
- **`IGSO3`**: Caches numerical approximations
  - Precomputes CDFs for inverse sampling
  - Stores score norms for efficient lookup
  - `sample_angle()`: Inverse CDF sampling
  - `score()`: Fast score lookup

- **`f_igso3()`**: IGSO(3) density function (truncated series)
- **`calculate_igso3()`**: Precomputation over discretized t and ω grid

**Interactions**:
- Used by `so3_diffuser.py` for efficient sampling and scoring
- Caches results to disk to avoid recomputation

---

### Geometric Utilities

#### `all_atom.py` (222 lines)
**Purpose**: Converts between backbone representations and all-atom structures.

**Key Components**:
- **`torsion_angles_to_frames()`**: Converts torsion angles → 8 rigid frames per residue
- **`frames_to_atom14_pos()`**: Converts frames → idealized atom14 positions
- **`compute_backbone()`**: Generates backbone atoms from rigid transforms + psi angles
- **`calculate_neighbor_angles()`**: Computes angles between vectors
- **`vector_projection()`**: Projects vectors onto planes

**Interactions**:
- Used by `score_network.py` to compute atom positions from predictions
- Used by `ipa_pytorch.py` for geometric calculations
- Relies on OpenFold's `rigid_utils` and `residue_constants`

---

#### `so3_utils.py` (Not viewed but referenced)
**Purpose**: SO(3) utility functions (axis-angle, rotation matrices, etc.)

---

### Data Loading

#### `pdb_data_loader.py` (503 lines)
**Purpose**: PyTorch dataset for loading protein structures from PDB files.

**Key Components**:
- **`PdbDataset`**: Main dataset class
  - Reads preprocessed PDB features from disk
  - Applies diffusion to create training pairs (clean, noised)
  - Handles fixed/diffused residue masking
  - Returns batched features for training

- **`TrainSampler`**: Custom sampler for length-based batching
  - Groups proteins by length for efficient batching
  - Supports cluster-based sampling to avoid overfitting

- **`DistributedTrainSampler`**: Multi-GPU version of TrainSampler

**Interactions**:
- Used by `train_se3_diffusion.py` to create data loaders
- Uses `se3_diffuser.py` to noise structures
- Reads files processed by `process_pdb_dataset.py`

---

#### `protein.py` (279 lines)
**Purpose**: Protein data structure (from AlphaFold/OpenFold).

**Key Components**:
- **`Protein`**: Dataclass holding protein structure
  - `atom_positions`: [N, 37, 3] all-atom coordinates
  - `aatype`: [N] amino acid types
  - `atom_mask`: [N, 37] which atoms are present
  
- **`from_pdb_string()`**: Parses PDB format → Protein object
- **`to_pdb()`**: Converts Protein → PDB format string

**Interactions**:
- Used throughout for representing protein structures
- Used by `pdb_data_loader.py` and inference scripts

---

#### `utils.py` (616 lines)
**Purpose**: Miscellaneous utilities for data processing.

**Key Components**:
- **`parse_pdb_feats()`**: Parses PDB files into feature dictionaries
- **`parse_chain_feats()`**: Processes chain-level features
- **`rigid_frames_from_all_atom()`**: Extracts rigid frames from atom positions
- **`pad_feats()`**: Pads features to maximum length
- **`create_data_loader()`**: Creates PyTorch DataLoader
- **`write_checkpoint()` / `read_pkl()`**: Checkpoint I/O

**Interactions**:
- Used by data loaders, training, and inference scripts
- Provides glue code between different representations

---

### Constants and Parsing

#### `residue_constants.py` (Not viewed, ~35KB)
**Purpose**: Amino acid constants from AlphaFold/OpenFold.
- Atom names, ideal geometries, torsion angle definitions, etc.

#### `chemical.py` (572 lines)
**Purpose**: Chemical constants for amino acids.
- Atom mappings (aa2long, aa2longalt)
- Bond definitions (aabonds)
- Amino acid type conversions

#### `mmcif_parsing.py`, `parsers.py`, `process_pdb_files.py`
**Purpose**: PDB/mmCIF file parsing utilities.

#### `errors.py`
**Purpose**: Custom exception classes.

---

## 3. `experiments/` Directory

Contains scripts for training and inference.

### `train_se3_diffusion.py` (834 lines)
**Purpose**: Main training script using Hydra configuration.

**Key Components**:
- **`Experiment`**: Main training class
  - `create_dataset()`: Sets up PdbDataset and data loaders
  - `loss_fn()`: Computes diffusion loss (MSE on scores)
  - `update_fn()`: Single training step
  - `train_epoch()`: Full epoch of training + validation
  - `inference_fn()`: Reverse diffusion sampling
  - `start_training()`: Main training loop

**Training Process**:
1. Sample time t uniformly
2. Noise clean structure using `diffuser.forward_marginal()`
3. Predict scores using `model()`
4. Compute loss vs. ground truth scores
5. Backpropagate and update weights

**Interactions**:
- Uses `PdbDataset` from `pdb_data_loader.py`
- Uses `ScoreNetwork` from `score_network.py`
- Uses `SE3Diffuser` from `se3_diffuser.py`
- Logs to Weights & Biases (wandb)

---

### `inference_se3_diffusion.py` (474 lines)
**Purpose**: Sampling and inference from trained models.

**Key Components**:
- **`Sampler`**: Inference class
  - `_load_ckpt()`: Loads trained checkpoint
  - `sample()`: Generates structures of specified length
  - `run_sampling()`: Runs full sampling pipeline
  - `save_traj()`: Saves trajectory as PDB files
  - `run_self_consistency()`: Evaluates designs using ESMFold

**Sampling Process**:
1. Initialize from reference distribution (noise)
2. Iteratively denoise using `diffuser.reverse()` and model predictions
3. Save intermediate states and final structure

**Interactions**:
- Reuses `Experiment` class from `train_se3_diffusion.py`
- Uses trained `ScoreNetwork` model
- Outputs PDB files for visualization

---

### `utils.py` (113 lines)
**Purpose**: Experiment utilities.

**Key Components**:
- **`get_ddp_info()`**: Distributed training info
- **`flatten_dict()`**: Flattens nested config dicts
- **`t_stratified_loss()`**: Bins loss by timestep for analysis
- **`get_sampled_mask()`**: Parses contig strings for motif scaffolding

**Interactions**:
- Used by training and inference scripts
- Helps with distributed training and logging

---

## Data Flow

### Training Flow
```
PDB files → PdbDataset → DataLoader
                ↓
    (clean structure, t) → SE3Diffuser.forward_marginal() → noised structure
                ↓
    noised structure → ScoreNetwork → predicted scores
                ↓
    predicted scores vs. ground truth scores → Loss → Backprop
```

### Inference Flow
```
Random noise (t=1) → ScoreNetwork → predicted scores
                ↓
    SE3Diffuser.reverse() → slightly denoised structure (t-dt)
                ↓
    Repeat until t=0 → Final structure → Save as PDB
```

---

## Key Interactions Between Directories

1. **`model/` ← `data_se3/`**:
   - `score_network.py` uses `se3_diffuser.py` to compute ground truth scores
   - `ipa_pytorch.py` uses `all_atom.py` for geometric operations

2. **`experiments/` ← `model/` + `data_se3/`**:
   - Training script instantiates `ScoreNetwork` and `SE3Diffuser`
   - Uses `PdbDataset` for data loading
   - Uses diffuser for noising and denoising

3. **Diffusion Hierarchy**:
   ```
   SE3Diffuser (se3_diffuser.py)
        ├── SO3Diffuser (so3_diffuser.py) → IGSO3 (igso3.py)
        └── R3Diffuser (r3_diffuser.py)
   ```

4. **Model Hierarchy**:
   ```
   ScoreNetwork (score_network.py)
        ├── Embedder
        └── IpaScore (ipa_pytorch.py)
             ├── InvariantPointAttention
             ├── BackboneUpdate
             └── TorsionAngles
   ```

---

## Summary

This codebase implements a complete SE(3) diffusion pipeline for protein structure generation:

- **`data_se3/`** provides the mathematical foundation (diffusion processes, geometric operations, data loading)
- **`model/`** implements the neural network that learns to predict denoising scores
- **`experiments/`** ties everything together for training and inference

The key innovation is using diffusion on the SE(3) group to generate protein backbones while respecting 3D geometric constraints through Invariant Point Attention.
