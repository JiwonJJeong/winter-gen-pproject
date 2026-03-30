"""Unconditional SE(3) score-matching diffusion training.

Follows the SinFusion training pattern:
  - SE3Diffusion.forward() owns t-sampling, SE3 forward marginal, model, and loss
  - CosineAnnealingLR schedule
  - max_steps instead of max_epochs

References:
  - SinFusion: single-trajectory DDPM with step-based training, cosine schedule
  - MDGen: gradient clipping, EMA, num_workers, accumulate_grad_batches

Usage:
    python gen_model/train_unconditional.py --data_dir data
"""
import torch
import os
import argparse
import pytorch_lightning as L

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from gen_model.train_base import default_se3_conf, default_model_conf, default_data_args
from gen_model.se3_diffusion_module import SE3Diffusion


def main():
    parser = argparse.ArgumentParser(description='Unconditional SE(3) Diffusion Training')
    parser.add_argument('--data_dir',    type=str,   default='data')
    parser.add_argument('--atlas_csv',   type=str,   default='gen_model/splits/atlas.csv')
    parser.add_argument('--suffix',      type=str,   default='_latent')
    parser.add_argument('--protein',     type=str,   required=True,
                        help='Protein name to train on (single trajectory, SinFusion-style)')
    parser.add_argument('--replica',     type=str,   default='1',
                        help='Replica suffix (e.g. "1" for _R1); single-trajectory training uses one replica')
    parser.add_argument('--batch_size',  type=int,   default=1,
                        help='Batch size (SinFusion default: 1 for single-trajectory training)')
    parser.add_argument('--max_steps',   type=int,   default=200_000,
                        help='Total training steps (SinFusion-style step-based training)')
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--save_dir',    type=str,   default='checkpoints/unconditional')
    parser.add_argument('--lora_r',      type=int,   default=0)
    parser.add_argument('--lora_alpha',  type=float, default=0.0)
    parser.add_argument('--num_blocks',  type=int,   default=8,
                        help='Number of IPA blocks (paper: 8; Colab: 4 for speed)')
    # MDGen-inspired options
    parser.add_argument('--grad_clip',   type=float, default=1.0,
                        help='Gradient clipping value (MDGen default: 1.0)')
    parser.add_argument('--ema_decay',   type=float, default=0.999,
                        help='EMA decay rate (0 = disabled, MDGen default: 0.999)')
    parser.add_argument('--accumulate_grad', type=int, default=1,
                        help='Accumulate gradients over N batches (MDGen-style)')
    parser.add_argument('--num_workers', type=int,   default=4,
                        help='DataLoader workers')
    parser.add_argument('--virtual_epoch_size', type=int, default=5000,
                        help='Virtual epoch size (SinFusion default: 5000; 0 = use real dataset size)')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Resume from checkpoint (restores weights, optimizer, step counter)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    from gen_model.diffusion.se3_diffuser import SE3Diffuser
    from gen_model.models.star_score_network import StarScoreNetwork
    from gen_model.models.lora import apply_lora
    from gen_model.data.dataset import MDGenDataset

    se3_conf   = default_se3_conf()
    # star.enabled=False → StarScoreNetwork behaves identically to ScoreNetwork
    model_conf = default_model_conf(
        star_enabled=False,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        num_blocks=args.num_blocks,
    )
    data_args  = default_data_args(args)
    diffuser   = SE3Diffuser(se3_conf)

    # Dataset — use all frames (SinFusion-style: no train/val split within one trajectory).
    # Evaluation is done post-hoc against held-out replicas via evaluate.py.
    train_dataset = MDGenDataset(
        args=data_args, mode='all',
        virtual_epoch_size=args.virtual_epoch_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    # Build model and wrap in SinFusion-style Lightning module
    score_network = StarScoreNetwork(model_conf, diffuser)
    apply_lora(score_network, model_conf.lora)

    module = SE3Diffusion(
        model=score_network,
        diffuser=diffuser,
        lr=args.lr,
        cosine_T_max=args.max_steps,
        ema_decay=args.ema_decay,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='uncond-{step:06d}-{train_loss:.4f}',
        save_top_k=3, monitor='train_loss', mode='min', save_last=True,
    )
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator='auto', devices='auto',
        callbacks=[checkpoint_cb],
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=100,
        enable_progress_bar=True,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad,
    )
    trainer.fit(module, train_dataloaders=train_loader, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
