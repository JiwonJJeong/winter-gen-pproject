"""SE(3) diffusion training with PyTorch Lightning.

Trains a ScoreNetwork (IPA transformer) on MD trajectory rigid-body frames
using SE(3) score-matching: the dataset applies the SE(3) forward process
(IGSO3 rotation + VP-SDE translation) and the model learns to predict the
reverse scores.

Usage:
    python gen_model/simple_train.py --data_dir data --save_dir checkpoints/se3
"""
import torch
import torch.nn.functional as F
import os
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


class SE3Module(L.LightningModule):
    """Lightning module for SE(3) score-matching diffusion on protein backbones.

    The dataset must be initialised with an SE3Diffuser so that every batch
    already contains rigids_t, rot_score, and trans_score from the forward
    diffusion process. This module predicts those scores and minimises a
    scale-normalised MSE loss.
    """

    def __init__(self, model_conf, se3_conf, lr=1e-4, rot_loss_weight=1.0, trans_loss_weight=1.0, psi_loss_weight=1.0):
        """Args:
            model_conf: Score network config (node_embed_size, edge_embed_size, embed, ipa).
            se3_conf: SE3Diffuser config (diffuse_rot, diffuse_trans, so3, r3).
            lr: Learning rate.
            rot_loss_weight: Weight for rotation score loss.
            trans_loss_weight: Weight for translation score loss.
            psi_loss_weight: Weight for psi torsion angle loss.
        """
        super().__init__()
        # Only save primitive hparams; DictConfig objects are not Lightning-serialisable
        self.save_hyperparameters(ignore=['model_conf', 'se3_conf'])
        from gen_model.diffusion.se3_diffuser import SE3Diffuser
        from gen_model.models.score_network import ScoreNetwork
        diffuser = SE3Diffuser(se3_conf)
        self.model = ScoreNetwork(model_conf, diffuser)
        self.lr = lr
        self.rot_loss_weight = rot_loss_weight
        self.trans_loss_weight = trans_loss_weight
        self.psi_loss_weight = psi_loss_weight

    def _compute_loss(self, batch):
        pred = self.model(batch)

        mask = batch['res_mask'].float()                           # [B, N]
        rot_score = batch['rot_score'].float()                     # [B, N, 3]
        trans_score = batch['trans_score'].float()                 # [B, N, 3]
        rot_score_scaling = batch['rot_score_scaling'].float()     # [B]
        trans_score_scaling = batch['trans_score_scaling'].float() # [B]

        # Reshape scaling for broadcasting: [B] → [B, 1, 1]
        rot_scaling = rot_score_scaling[:, None, None] + 1e-8
        trans_scaling = trans_score_scaling[:, None, None] + 1e-8

        # Scale-normalised MSE equalises loss magnitude across timesteps
        rot_mse = F.mse_loss(
            pred['rot_score'] / rot_scaling,
            rot_score / rot_scaling,
            reduction='none',
        ).sum(dim=-1)   # [B, N]
        trans_mse = F.mse_loss(
            pred['trans_score'] / trans_scaling,
            trans_score / trans_scaling,
            reduction='none',
        ).sum(dim=-1)   # [B, N]

        n_visible = mask.sum() + 1e-8
        rot_loss = (rot_mse * mask).sum() / n_visible
        trans_loss = (trans_mse * mask).sum() / n_visible

        # Psi torsion loss: pred['psi'] is [B, N, 2] (sin/cos), already normalised
        # by ScoreNetwork. Ground truth is torsion_angles_sin_cos[..., 2, :] (index 2 = psi).
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :].float()  # [B, N, 2]
        psi_mse = F.mse_loss(pred['psi'], gt_psi, reduction='none').sum(dim=-1)  # [B, N]
        psi_loss = (psi_mse * mask).sum() / n_visible

        total = (self.rot_loss_weight * rot_loss
                 + self.trans_loss_weight * trans_loss
                 + self.psi_loss_weight * psi_loss)
        return total, rot_loss, trans_loss, psi_loss

    def training_step(self, batch, batch_idx):
        loss, rot_loss, trans_loss, psi_loss = self._compute_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_rot_loss', rot_loss, on_step=False, on_epoch=True)
        self.log('train_trans_loss', trans_loss, on_step=False, on_epoch=True)
        self.log('train_psi_loss', psi_loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, rot_loss, trans_loss, psi_loss = self._compute_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_rot_loss', rot_loss, on_step=False, on_epoch=True)
        self.log('val_trans_loss', trans_loss, on_step=False, on_epoch=True)
        self.log('val_psi_loss', psi_loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    """CLI entry point with sensible SE(3) defaults."""
    parser = argparse.ArgumentParser(description='SE(3) Diffusion Training')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--atlas_csv', type=str, default='gen_model/splits/atlas.csv')
    parser.add_argument('--train_split', type=str, default='gen_model/splits/frame_splits.csv')
    parser.add_argument('--suffix', type=str, default='_latent')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints/se3')
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
    val_dataset = MDGenDataset(args=data_args, diffuser=diffuser, mode='val')
    val_dataset.coord_scale = float(train_dataset.coord_scale)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

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
