"""Simple DDPM training script for noising/denoising single frames.

This script implements basic DDPM training where we:
1. Load a single frame from MD trajectory data
2. Add noise to it according to a diffusion schedule
3. Train a model to predict the original frame (or the noise)
4. Save checkpoints periodically

Usage:
    python gen_model/simple_train.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm


class SimpleDDPM:
    """Basic DDPM implementation for frame denoising."""

    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """Initialize DDPM noise schedule.

        Args:
            timesteps: Number of diffusion timesteps
            beta_start: Starting beta value (small noise)
            beta_end: Ending beta value (large noise)
        """
        self.timesteps = timesteps

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute values for forward and reverse diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior variance for reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_0, t, noise=None):
        """Forward diffusion: add noise to data.

        Args:
            x_0: Original data [B, ...]
            t: Timestep(s) [B] or scalar
            noise: Optional pre-generated noise

        Returns:
            Noisy data x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[..., None]
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[..., None]

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def to(self, device):
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


class SimpleDenoiseModel(nn.Module):
    """Simple U-Net-like model for denoising frames.

    Takes noisy frame and timestep, outputs predicted noise or clean frame.
    """

    def __init__(self, in_channels, hidden_dim=128, time_emb_dim=128):
        """Initialize model.

        Args:
            in_channels: Total number of input channels (e.g., N_res * N_atoms * 3 for N_res residues)
            hidden_dim: Hidden dimension size
            time_emb_dim: Timestep embedding dimension
        """
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Simple encoder-decoder architecture
        self.enc1 = nn.Sequential(
            nn.Linear(in_channels + time_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.SiLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.SiLU(),
        )

        # Decoder with skip connections
        self.dec3 = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Output projection
        self.out = nn.Linear(hidden_dim, in_channels)

    def forward(self, x, t):
        """Forward pass.

        Args:
            x: Noisy input [B, N, C] where N is number of residues/atoms
            t: Timesteps [B]

        Returns:
            Predicted noise or clean frame [B, N, C]
        """
        batch_size, n_residues, channels = x.shape

        # Flatten spatial dimensions
        x_flat = x.reshape(batch_size, -1)  # [B, N*C]

        # Time embedding
        t_emb = self.time_mlp(t.float().unsqueeze(-1) / 1000.0)  # Normalize timestep

        # Concatenate time embedding with input
        x_flat = torch.cat([x_flat, t_emb], dim=-1)

        # Encoder with skip connections
        e1 = self.enc1(x_flat)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([b, e3], dim=-1))
        d2 = self.dec2(torch.cat([d3, e2], dim=-1))
        d1 = self.dec1(torch.cat([d2, e1], dim=-1))

        # Output
        out = self.out(d1)

        # Reshape back
        out = out.reshape(batch_size, n_residues, channels)

        return out


def train_ddpm(
    dataset,
    model,
    diffusion,
    device='cuda',
    batch_size=4,
    num_epochs=100,
    lr=1e-4,
    save_dir='checkpoints/simple_ddpm',
    save_every=10
):
    """Train DDPM model with spatial masking support.

    Spatial Masking:
        - If dataset provides 'res_mask' or 'mask', applies spatial masking
        - Masked residues (mask=0) are replaced with random noise (not visible to model)
        - Loss is computed only on visible residues (mask=1)
        - This improves generalization and removes edge bias in training

    Args:
        dataset: Dataset instance (should provide 'res_mask' if using spatial masking)
        model: Denoising model
        diffusion: SimpleDDPM instance
        device: Device to train on
        batch_size: Batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
    """
    os.makedirs(save_dir, exist_ok=True)

    # Move model and diffusion to device
    model = model.to(device)
    diffusion = diffusion.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, batch in enumerate(progress_bar):
            # Get frame data (assuming batch contains 'atom14_pos' or similar)
            # Shape: [B, N_res, N_atoms, 3] or [B, N_res*N_atoms*3]
            if 'atom14_pos' in batch:
                x_0 = batch['atom14_pos'].to(device)
            elif 'rigids_0' in batch:
                # Extract translations from rigids
                x_0 = batch['rigids_0'][..., 4:].to(device)
            else:
                raise ValueError("Batch must contain 'atom14_pos' or 'rigids_0'")

            batch_size_actual = x_0.shape[0]

            # Flatten to [B, N, C] format if needed
            if len(x_0.shape) == 4:
                x_0 = x_0.reshape(batch_size_actual, x_0.shape[1], -1)

            # Get spatial mask (1 = visible, 0 = masked)
            if 'res_mask' in batch:
                mask = batch['res_mask'].to(device)  # [B, L]
            elif 'mask' in batch:
                mask = batch['mask'].to(device)  # [B, L]
            else:
                # No mask provided - use all residues
                mask = torch.ones(batch_size_actual, x_0.shape[1], device=device)

            mask_expanded = mask.unsqueeze(-1)  # [B, L, 1] for broadcasting

            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size_actual,), device=device)

            # Generate noise for visible residues
            noise = torch.randn_like(x_0)

            # Add noise to get x_t (for visible residues)
            x_t = diffusion.add_noise(x_0, t, noise)

            # Replace masked residues with independent random noise
            # This prevents the model from using masked residues as information
            noise_masked = torch.randn_like(x_0) * (1 - mask_expanded)
            x_t = mask_expanded * x_t + noise_masked

            # Predict noise (model doesn't know which residues are masked)
            pred_noise = model(x_t, t)

            # Compute loss ONLY on visible residues
            loss_per_element = F.mse_loss(pred_noise, noise, reduction='none')
            masked_loss = loss_per_element * mask_expanded

            # Normalize by number of visible elements
            num_visible = mask.sum() * x_0.shape[-1] + 1e-8
            loss = masked_loss.sum() / num_visible

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}')

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')


def main():
    """Main training function."""
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Create dataset
    from gen_model.dataset import MDGenDataset
    from omegaconf import OmegaConf

    args = OmegaConf.create({
        'data_dir': 'data',
        'atlas_csv': 'data/atlas.csv',
        'train_split': 'gen_model/splits/frame_splits.csv',
        'suffix': '_latent',
        'frame_interval': None,
        'crop_ratio': 0.95,
        'min_t': 0.01,
    })

    dataset = MDGenDataset(
        args=args,
        diffuser=None,  # No SE3 diffuser needed for basic DDPM
        split=args.train_split,
        mode='train',
        repeat=1,
        num_consecutive=1,
        stride=1
    )

    print(f'Dataset size: {len(dataset)}')

    # Get sample to determine input dimensions
    sample = dataset[0]
    if 'atom14_pos' in sample:
        sample_data = sample['atom14_pos']
    elif 'rigids_0' in sample:
        sample_data = sample['rigids_0'][..., 4:]
    else:
        raise ValueError("Sample must contain 'atom14_pos' or 'rigids_0'")

    if len(sample_data.shape) == 3:
        # [N_res, N_atoms, 3]
        n_residues = sample_data.shape[0]
        in_channels = n_residues * sample_data.shape[1] * sample_data.shape[2]
    else:
        # [N_res, 3] or [N_res, C]
        n_residues = sample_data.shape[0]
        in_channels = n_residues * sample_data.shape[1]

    print(f'Input shape: {sample_data.shape}')
    print(f'Flattened input channels: {in_channels}')

    # Create model and diffusion
    # Use total flattened dimension: in_channels + time_emb_dim
    model = SimpleDenoiseModel(in_channels=in_channels, hidden_dim=256, time_emb_dim=128)
    diffusion = SimpleDDPM(timesteps=1000, beta_start=0.0001, beta_end=0.02)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    # Train
    train_ddpm(
        dataset=dataset,
        model=model,
        diffusion=diffusion,
        device=device,
        batch_size=8,
        num_epochs=100,
        lr=1e-4,
        save_dir='checkpoints/simple_ddpm',
        save_every=10
    )

    print('Training complete!')


if __name__ == '__main__':
    main()
