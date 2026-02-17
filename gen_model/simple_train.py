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
import numpy as np
import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


# Base classes from original code
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


class DDPMModule(L.LightningModule):
    def __init__(self, in_channels, lr=1e-4, timesteps=1000, coord_scale=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleDenoiseModel(in_channels=in_channels)
        self.diffusion = SimpleDDPM(timesteps=timesteps)
        self.lr = lr
        self.coord_scale = coord_scale

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        if 'atom14_pos' in batch:
            x_0 = batch['atom14_pos']
        elif 'rigids_0' in batch:
            x_0 = batch['rigids_0'][..., 4:]
        else:
            raise ValueError("Batch must contain 'atom14_pos' or 'rigids_0'")

        # Flatten if needed
        # Expected model input is [B, N, C]
        if len(x_0.shape) == 4:
            # [B, N_res, N_atoms, 3] -> [B, N_res, 14*3]
            x_0 = x_0.reshape(x_0.shape[0], x_0.shape[1], -1)

        batch_size, n_residues, channels = x_0.shape

        # Sample noise and timesteps
        t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(x_0)

        # Add noise
        x_t = self.diffusion.add_noise(x_0, t, noise)

        # Restore Masked Residue Logic
        if 'res_mask' in batch or 'mask' in batch:
            mask = batch.get('res_mask', batch.get('mask'))
            mask_expanded = mask.unsqueeze(-1) # [B, N, 1]
            
            # Replace masked residues with independent random noise
            noise_masked = torch.randn_like(x_0) * (1 - mask_expanded)
            x_t = mask_expanded * x_t + noise_masked
            
            # Predict noise
            pred_noise = self.model(x_t, t)
            
            # Compute loss only on visible residues
            loss_per_element = F.mse_loss(pred_noise, noise, reduction='none')
            masked_loss = loss_per_element * mask_expanded
            
            # Normalize by number of visible elements
            num_visible = mask.sum() * channels + 1e-8
            loss = masked_loss.sum() / num_visible
        else:
            # No mask, standard MSE
            pred_noise = self.model(x_t, t)
            loss = F.mse_loss(noise, pred_noise)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='DDPM Training with Lightning')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--atlas_csv', type=str, default='data/atlas.csv', help='Atlas CSV file')
    parser.add_argument('--train_split', type=str, default='gen_model/splits/frame_splits.csv', help='Split file')
    parser.add_argument('--suffix', type=str, default='_latent', help='Suffix for npy files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints/simple_ddpm', help='Save directory')
    parser.add_argument('--timesteps', type=int, default=1000, help='Diffusion timesteps')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Initialize DataModule
    from gen_model.dataset import MDGenDataModule
    datamodule = MDGenDataModule(args, batch_size=args.batch_size)
    datamodule.setup() 

    # 2. Determine in_channels from a sample
    sample = datamodule.train_dataset[0]
    if 'atom14_pos' in sample:
        sample_data = sample['atom14_pos']
    else:
        sample_data = sample['rigids_0'][..., 4:]

    if len(sample_data.shape) == 3:
        # [N_res, N_atoms, 3] -> total flattened
        n_residues = sample_data.shape[0]
        in_channels = n_residues * sample_data.shape[1] * sample_data.shape[2]
    else:
        n_residues = sample_data.shape[0]
        in_channels = n_residues * sample_data.shape[1]

    # 3. Initialize Model
    model = DDPMModule(
        in_channels=in_channels,
        lr=args.lr,
        timesteps=args.timesteps,
        coord_scale=datamodule.coord_scale
    )

    # 4. Set up Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='ddpm-{epoch:02d}-{train_loss:.4f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min',
        save_last=True
    )

    # 5. Training
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()
