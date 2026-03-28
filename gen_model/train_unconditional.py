"""Unconditional SE(3) score-matching diffusion training.

Follows the SinFusion training pattern:
  - SE3Diffusion.forward() owns t-sampling, SE3 forward marginal, model, and loss
  - CosineAnnealingLR schedule
  - max_steps instead of max_epochs

Usage:
    python gen_model/train_unconditional.py --data_dir data
"""
import torch
import os
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from gen_model.train_base import default_se3_conf, default_model_conf, default_data_args
from gen_model.se3_diffusion_module import SE3Diffusion


def main():
    parser = argparse.ArgumentParser(description='Unconditional SE(3) Diffusion Training')
    parser.add_argument('--data_dir',    type=str,   default='data')
    parser.add_argument('--atlas_csv',   type=str,   default='gen_model/splits/atlas.csv')
    parser.add_argument('--train_split', type=str,   default='gen_model/splits/frame_splits.csv')
    parser.add_argument('--suffix',      type=str,   default='_latent')
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--max_steps',   type=int,   default=200_000,
                        help='Total training steps (SinFusion-style step-based training)')
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--save_dir',    type=str,   default='checkpoints/unconditional')
    parser.add_argument('--lora_r',      type=int,   default=0)
    parser.add_argument('--lora_alpha',  type=float, default=0.0)
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
    )
    data_args  = default_data_args(args)
    diffuser   = SE3Diffuser(se3_conf)

    # Datasets — no diffuser passed: noise is applied in SE3Diffusion.forward()
    train_dataset = MDGenDataset(args=data_args, mode='train')
    val_dataset   = MDGenDataset(args=data_args, mode='val')
    val_dataset.coord_scale = float(train_dataset.coord_scale)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    # Build model and wrap in SinFusion-style Lightning module
    score_network = StarScoreNetwork(model_conf, diffuser)
    apply_lora(score_network, model_conf.lora)

    module = SE3Diffusion(
        model=score_network,
        diffuser=diffuser,
        lr=args.lr,
        cosine_T_max=args.max_steps,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='uncond-{step:06d}-{val_loss:.4f}',
        save_top_k=3, monitor='val_loss', mode='min', save_last=True,
    )
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator='auto', devices='auto',
        callbacks=[checkpoint_cb],
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
