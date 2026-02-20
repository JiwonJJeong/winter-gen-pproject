"""Unconditional SE(3) score-matching diffusion training.

The model learns the marginal distribution p(x) over protein backbone conformations.
Each training sample is a single trajectory frame noised by the SE(3) forward process;
the model predicts the reverse scores.

Usage:
    python gen_model/train_unconditional.py --data_dir data --save_dir checkpoints/se3
"""
import torch
import os
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from gen_model.train_base import SE3BaseModule, default_se3_conf, default_model_conf, default_data_args


class SE3Module(SE3BaseModule):
    """Unconditional SE(3) score-matching module.

    Wraps ScoreNetwork (IPA transformer) + SE3Diffuser. The batch is used
    as-is from MDGenDataset (sc_ca_t = zeros, single frame per sample).
    """

    def __init__(self, model_conf, se3_conf, lr=1e-4,
                 rot_loss_weight=1.0, trans_loss_weight=1.0, psi_loss_weight=1.0):
        super().__init__(lr=lr, rot_loss_weight=rot_loss_weight,
                         trans_loss_weight=trans_loss_weight,
                         psi_loss_weight=psi_loss_weight)
        self.save_hyperparameters(ignore=['model_conf', 'se3_conf'])
        from gen_model.diffusion.se3_diffuser import SE3Diffuser
        from gen_model.models.score_network import ScoreNetwork
        from gen_model.models.lora import apply_lora
        self.model = ScoreNetwork(model_conf, SE3Diffuser(se3_conf))
        apply_lora(self.model, model_conf.lora)


def main():
    """CLI entry point with sensible SE(3) defaults."""
    parser = argparse.ArgumentParser(description='Unconditional SE(3) Diffusion Training')
    parser.add_argument('--data_dir',    type=str,   default='data')
    parser.add_argument('--atlas_csv',   type=str,   default='gen_model/splits/atlas.csv')
    parser.add_argument('--train_split', type=str,   default='gen_model/splits/frame_splits.csv')
    parser.add_argument('--suffix',      type=str,   default='_latent')
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--save_dir',    type=str,   default='checkpoints/se3')
    parser.add_argument('--lora_r',     type=int,   default=0,
                        help='LoRA rank; 0 = disabled (full fine-tuning)')
    parser.add_argument('--lora_alpha', type=float, default=0.0,
                        help='LoRA alpha scaling; defaults to 2*lora_r when 0')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    from gen_model.data.dataset import MDGenDataset
    from gen_model.diffusion.se3_diffuser import SE3Diffuser

    se3_conf   = default_se3_conf()
    model_conf = default_model_conf(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    data_args  = default_data_args(args)
    diffuser   = SE3Diffuser(se3_conf)
    train_dataset = MDGenDataset(args=data_args, diffuser=diffuser, mode='train')
    val_dataset   = MDGenDataset(args=data_args, diffuser=diffuser, mode='val')
    val_dataset.coord_scale = float(train_dataset.coord_scale)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    model = SE3Module(model_conf=model_conf, se3_conf=se3_conf, lr=args.lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='se3-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3, monitor='val_loss', mode='min', save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator='auto', devices='auto',
        callbacks=[checkpoint_callback],
        precision='16-mixed' if torch.cuda.is_available() else 32,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
