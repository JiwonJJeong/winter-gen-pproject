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

from gen_model.train_base import SE3BaseModule


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
        self.model = ScoreNetwork(model_conf, SE3Diffuser(se3_conf))


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
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    from omegaconf import OmegaConf
    from gen_model.dataset import MDGenDataset
    from gen_model.diffusion.se3_diffuser import SE3Diffuser

    se3_conf = OmegaConf.create({
        'diffuse_rot': True, 'diffuse_trans': True,
        'so3': {'schedule': 'logarithmic', 'min_sigma': 0.1, 'max_sigma': 1.5,
                'num_sigma': 1000, 'use_cached_score': False,
                'cache_dir': '/tmp/igso3_cache', 'num_omega': 1000},
        'r3':  {'min_b': 0.1, 'max_b': 20.0, 'coordinate_scaling': 0.1},
    })
    model_conf = OmegaConf.create({
        'node_embed_size': 256, 'edge_embed_size': 128,
        'embed': {'index_embed_size': 32, 'embed_self_conditioning': True,
                  'num_bins': 22, 'min_bin': 1e-5, 'max_bin': 20.0},
        'ipa': {'c_s': 256, 'c_z': 128, 'c_hidden': 16, 'no_heads': 12,
                'no_qk_points': 4, 'no_v_points': 8, 'c_skip': 64,
                'num_blocks': 4, 'coordinate_scaling': 0.1,
                'seq_tfmr_num_heads': 4, 'seq_tfmr_num_layers': 2},
    })

    diffuser = SE3Diffuser(se3_conf)

    data_args = OmegaConf.create({
        'data_dir': args.data_dir, 'atlas_csv': args.atlas_csv,
        'train_split': args.train_split, 'suffix': args.suffix,
        'frame_interval': None, 'crop_ratio': 0.95, 'min_t': 0.01,
    })
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
