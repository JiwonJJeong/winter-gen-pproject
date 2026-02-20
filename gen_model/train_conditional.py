"""Conditional SE(3) score-matching diffusion training.

The model learns the transition kernel p(x_{t+k} | x_t) over protein backbone
conformations.  Each training sample pairs:
  - a clean SOURCE frame  → injected as sc_ca_t (CA distogram conditioning)
  - a noised TARGET frame → the object being denoised (k MD strides ahead/behind)

The signed integer gap k is embedded as ϕ(k) (sinusoidal, same size as ϕ(t))
and concatenated to the timestep embedding inside ScoreNetwork.Embedder
(enabled by embed.use_temporal_embedding = True in model_conf).

Curriculum learning (following the paper):
  - Epoch 0 .. grow_every-1      : k ∈ {-1, +1}
  - Epoch grow_every .. 2*ge-1   : k ∈ {-2, -1, +1, +2}
  - ...until current_max_k == max_k (default 3)
  TemporalCurriculumCallback updates current_max_k on both datasets each epoch.

Usage:
    python gen_model/train_conditional.py --data_dir data
    python gen_model/train_conditional.py --data_dir data --max_k 3 --grow_every 10
"""
import torch
import os
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from gen_model.train_base import SE3BaseModule, default_se3_conf, default_model_conf, default_data_args


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class TemporalCurriculumCallback(L.Callback):
    """Increases the temporal-gap curriculum range by 1 every `grow_every` epochs.

    Starts at current_max_k=1 (k ∈ {-1, +1}) and grows to max_k (default 3).
    Both the train and val datasets are updated at the start of each epoch.
    """

    def __init__(self, train_dataset, val_dataset, max_k: int = 3, grow_every: int = 10):
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.max_k         = max_k
        self.grow_every    = grow_every

    def on_train_epoch_start(self, trainer, pl_module):
        epoch   = trainer.current_epoch
        new_k   = min(1 + epoch // self.grow_every, self.max_k)
        self.train_dataset.current_max_k = new_k
        self.val_dataset.current_max_k   = new_k
        pl_module.log('current_max_k', float(new_k), on_epoch=True)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class ConditionalSE3Module(SE3BaseModule):
    """Conditional SE(3) score-matching module.

    Uses ScoreNetwork with use_temporal_embedding=True so that ϕ(k) is
    concatenated to ϕ(t) inside the Embedder.  The dataset supplies 'k' in
    every batch; the base loss is otherwise unchanged.
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for conditional SE(3) DDPM with temporal curriculum."""
    parser = argparse.ArgumentParser(description='Conditional SE(3) Diffusion Training')
    parser.add_argument('--data_dir',    type=str,   default='data')
    parser.add_argument('--atlas_csv',   type=str,   default='gen_model/splits/atlas.csv')
    parser.add_argument('--train_split', type=str,   default='gen_model/splits/frame_splits.csv')
    parser.add_argument('--suffix',      type=str,   default='_latent')
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--save_dir',    type=str,   default='checkpoints/conditional_se3')
    parser.add_argument('--stride',      type=int,   default=1,
                        help='Raw-trajectory stride between source and each k step')
    # Temporal curriculum args
    parser.add_argument('--max_k',       type=int,   default=3,
                        help='Maximum |k| sampled at full curriculum (paper: 3)')
    parser.add_argument('--grow_every',  type=int,   default=10,
                        help='Epochs between each +1 increase in current_max_k')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    from gen_model.data.dataset import ConditionalMDGenDataset
    from gen_model.diffusion.se3_diffuser import SE3Diffuser

    se3_conf   = default_se3_conf()
    model_conf = default_model_conf(use_temporal_embedding=True)  # enables ϕ(k)
    data_args  = default_data_args(args)
    diffuser   = SE3Diffuser(se3_conf)

    # Datasets: frame index built for full ±max_k range; curriculum starts at 1.
    train_dataset = ConditionalMDGenDataset(
        args=data_args, diffuser=diffuser, mode='train',
        stride=args.stride, max_k=args.max_k, current_max_k=1,
    )
    val_dataset = ConditionalMDGenDataset(
        args=data_args, diffuser=diffuser, mode='val',
        stride=args.stride, max_k=args.max_k, current_max_k=1,
    )
    val_dataset.coord_scale = float(train_dataset.coord_scale)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    model = ConditionalSE3Module(model_conf=model_conf, se3_conf=se3_conf, lr=args.lr)

    curriculum_callback = TemporalCurriculumCallback(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_k=args.max_k,
        grow_every=args.grow_every,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='cond-se3-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3, monitor='val_loss', mode='min', save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator='auto', devices='auto',
        callbacks=[curriculum_callback, checkpoint_callback],
        precision='16-mixed' if torch.cuda.is_available() else 32,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
