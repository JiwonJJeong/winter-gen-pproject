"""Simple DDPM inference script for denoising frames.

This script loads a trained DDPM model and performs inference by:
1. Starting from random noise (or a noisy frame)
2. Iteratively denoising using the reverse diffusion process
3. Saving the generated/denoised frame

Usage:
    python gen_model/simple_inference.py --checkpoint checkpoints/simple_ddpm/checkpoint_epoch_100.pt
"""
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm

from gen_model.simple_train import SimpleDDPM, SimpleDenoiseModel


@torch.no_grad()
def denoise_step(model, x_t, t, diffusion):
    """Perform single denoising step.

    Args:
        model: Denoising model
        x_t: Current noisy state [B, N, C]
        t: Current timestep(s) [B]
        diffusion: SimpleDDPM instance

    Returns:
        x_{t-1}: Less noisy state
    """
    batch_size = x_t.shape[0]

    # Predict noise
    pred_noise = model(x_t, t)

    # Get noise schedule values
    betas_t = diffusion.betas[t]
    sqrt_one_minus_alphas_cumprod_t = diffusion.sqrt_one_minus_alphas_cumprod[t]
    sqrt_recip_alphas_t = diffusion.sqrt_recip_alphas[t]

    # Reshape for broadcasting
    while len(betas_t.shape) < len(x_t.shape):
        betas_t = betas_t[..., None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[..., None]
        sqrt_recip_alphas_t = sqrt_recip_alphas_t[..., None]

    # Predict x_0 from x_t and predicted noise
    model_mean = sqrt_recip_alphas_t * (
        x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t[0] == 0:
        return model_mean
    else:
        posterior_variance_t = diffusion.posterior_variance[t]
        while len(posterior_variance_t.shape) < len(x_t.shape):
            posterior_variance_t = posterior_variance_t[..., None]

        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_from_noise(model, diffusion, shape, device='cuda', num_steps=None):
    """Generate samples starting from random noise.

    Args:
        model: Denoising model
        diffusion: SimpleDDPM instance
        shape: Shape of sample [B, N, C]
        device: Device to run on
        num_steps: Number of denoising steps (default: all timesteps)

    Returns:
        Generated sample
    """
    model.eval()

    if num_steps is None:
        num_steps = diffusion.timesteps

    # Start from random noise
    x_t = torch.randn(shape, device=device)

    # Reverse diffusion process
    timesteps = list(range(diffusion.timesteps))[:num_steps]
    for i in tqdm(reversed(timesteps), desc='Sampling', total=len(timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        x_t = denoise_step(model, x_t, t, diffusion)

    return x_t


@torch.no_grad()
def denoise_frame(model, diffusion, noisy_frame, num_steps=None, device='cuda'):
    """Denoise a specific noisy frame.

    Args:
        model: Denoising model
        diffusion: SimpleDDPM instance
        noisy_frame: Noisy input frame [B, N, C] or [N, C]
        num_steps: Number of denoising steps (default: all timesteps)
        device: Device to run on

    Returns:
        Denoised frame
    """
    model.eval()

    # Add batch dimension if needed
    if len(noisy_frame.shape) == 2:
        noisy_frame = noisy_frame.unsqueeze(0)

    noisy_frame = noisy_frame.to(device)

    if num_steps is None:
        num_steps = diffusion.timesteps

    x_t = noisy_frame

    # Reverse diffusion process
    timesteps = list(range(diffusion.timesteps))[:num_steps]
    for i in tqdm(reversed(timesteps), desc='Denoising', total=len(timesteps)):
        t = torch.full((noisy_frame.shape[0],), i, device=device, dtype=torch.long)
        x_t = denoise_step(model, x_t, t, diffusion)

    return x_t


def load_checkpoint(checkpoint_path, model, device='cuda'):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        device: Device to load on

    Returns:
        Loaded model, epoch number, loss
    """
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])    
    epoch = checkpoint['epoch']
    loss = checkpoint.get('loss', None)
    coord_scale = checkpoint.get('coord_scale', 0.1)

    print(f'Loaded checkpoint from epoch {epoch}')
    if loss is not None:
        print(f'Checkpoint loss: {loss:.6f}')
    print(f'Loaded coord_scale: {coord_scale:.4f}')

    return model, epoch, loss, coord_scale


def test_with_dataset(model, diffusion, dataset, device='cuda', coord_scale=0.1, num_samples=5):
    """Test model by noising and denoising frames from dataset.

    Args:
        model: Denoising model
        diffusion: SimpleDDPM instance
        dataset: Dataset instance
        device: Device to run on
        coord_scale: Scaling factor for unscaling
        num_samples: Number of samples to test

    Returns:
        List of (original, noisy, denoised) tuples
    """
    model.eval()
    results = []

    for i in range(min(num_samples, len(dataset))):
        # Get sample
        sample = dataset[i]

        if 'atom14_pos' in sample:
            x_0 = sample['atom14_pos']
        elif 'rigids_0' in sample:
            x_0 = sample['rigids_0'][..., 4:]
        else:
            raise ValueError("Sample must contain 'atom14_pos' or 'rigids_0'")

        # Flatten if needed
        if len(x_0.shape) == 3:
            x_0 = x_0.reshape(x_0.shape[0], -1)

        x_0 = x_0.unsqueeze(0).to(device)

        # Add noise at a specific timestep (e.g., middle of schedule)
        t = torch.tensor([diffusion.timesteps // 2], device=device)
        noise = torch.randn_like(x_0)
        x_noisy = diffusion.add_noise(x_0, t, noise)

        # Denoise
        x_denoised = denoise_frame(model, diffusion, x_noisy, device=device)

        # Convert to numpy
        # Move predictions to CPU and unscale
        # x_0 and x_pred are scaled. Reverse transform for MSE and saving.
        # Centroid is restored for full reconstruction if needed.
        centroid = sample['centroid'].unsqueeze(0).to(device) # [1, 3]

        x_0_unscaled = (x_0.squeeze(0).cpu() / coord_scale) + centroid.cpu()
        x_pred_unscaled = (x_denoised.squeeze(0).cpu() / coord_scale) + centroid.cpu()
        x_noisy_unscaled = (x_noisy.squeeze(0).cpu() / coord_scale) + centroid.cpu()

        mse = torch.mean((x_0_unscaled - x_pred_unscaled)**2)
        print(f'Sample {i+1}: reconstruction MSE = {mse.item():.6f} (Angstroms^2)')

        results.append({
            'original': x_0_unscaled.numpy(),
            'noisy': x_noisy_unscaled.numpy(),
            'denoised': x_pred_unscaled.numpy(),
            'mse': mse.item()
        })

    return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='DDPM Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--mode', type=str, default='sample', choices=['sample', 'denoise', 'test'],
                        help='Inference mode: sample from noise, denoise specific frame, or test on dataset')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples for test mode')
    parser.add_argument('--output_dir', type=str, default='outputs/simple_ddpm', help='Output directory')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model configuration from checkpoint or use default
    # For now, using default configuration - should match training
    from gen_model.dataset import MDGenDataset
    from omegaconf import OmegaConf

    dataset_args = OmegaConf.create({
        'data_dir': 'data',
        'atlas_csv': 'data/atlas.csv',
        'train_split': 'gen_model/splits/frame_splits.csv',
        'suffix': '_latent',
        'frame_interval': None,
        'crop_ratio': 0.95,
        'min_t': 0.01,
    })

    dataset = MDGenDataset(
        args=dataset_args,
        diffuser=None,
        split=dataset_args.train_split,
        mode='val',
        repeat=1,
        num_consecutive=1,
        stride=1
    )
    args.dataset = dataset # Attach dataset to args for test_with_dataset

    # Get sample to determine dimensions
    sample = dataset[0]
    if 'atom14_pos' in sample:
        sample_data = sample['atom14_pos']
    elif 'rigids_0' in sample:
        sample_data = sample['rigids_0'][..., 4:]
    else:
        raise ValueError("Sample must contain 'atom14_pos' or 'rigids_0'")

    if len(sample_data.shape) == 3:
        n_residues = sample_data.shape[0]
        in_channels = n_residues * sample_data.shape[1] * sample_data.shape[2]
    else:
        n_residues = sample_data.shape[0]
        in_channels = n_residues * sample_data.shape[1]

    # Create model and diffusion
    # Use total flattened dimension: in_channels + time_emb_dim
    # model = SimpleDenoiseModel(in_channels=in_channels, hidden_dim=256, time_emb_dim=128) # Original line
    # diffusion = SimpleDDPM(timesteps=1000, beta_start=0.0001, beta_end=0.02) # Original line
    # Load model architecture and checkpoint
    model = SimpleDenoiseModel(in_channels=in_channels, hidden_dim=256, time_emb_dim=128) # Re-added hidden_dim and time_emb_dim
    diffusion = SimpleDDPM(timesteps=1000, beta_start=0.0001, beta_end=0.02) # Re-added diffusion init
    model, epoch, loss, coord_scale = load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    
    # Run inference based on mode
    if args.mode == 'sample':
        print('Sampling from random noise...')
        shape = (1, n_residues, in_channels)
        samples = sample_from_noise(model, diffusion, shape, device)

        # Save samples
        output_path = os.path.join(args.output_dir, 'generated_sample.npy')
        np.save(output_path, samples.cpu().numpy())
        print(f'Saved generated sample to {output_path}')

    elif args.mode == 'test':
        print(f'Testing on {args.num_samples} samples from dataset...')
        results = test_with_dataset(model, diffusion, dataset, device, coord_scale, args.num_samples)

        # Save results
        for i, result in enumerate(results):
            np.save(os.path.join(args.output_dir, f'test_{i}_original.npy'), result['original'])
            np.save(os.path.join(args.output_dir, f'test_{i}_noisy.npy'), result['noisy'])
            np.save(os.path.join(args.output_dir, f'test_{i}_denoised.npy'), result['denoised'])

        avg_mse = np.mean([r['mse'] for r in results])
        print(f'\nAverage MSE: {avg_mse:.6f}')
        print(f'Results saved to {args.output_dir}')

    else:  # denoise mode
        print('Denoise mode not yet implemented. Use --mode test instead.')


if __name__ == '__main__':
    main()
