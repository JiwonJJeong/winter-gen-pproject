"""Conditional SE(3) score-matching diffusion training.

Follows the SinFusion training pattern (ConditionalDiffusion analog):
  - ConditionalSE3Diffusion.forward() owns t-sampling, SE3 forward marginal,
    model, and loss — sc_ca_t (source CA) conditions the model, analogous to
    SinFusion's CONDITION_IMG; k (temporal stride) analogous to FRAME.
  - CosineAnnealingLR schedule
  - max_steps instead of max_epochs
  - Temporal curriculum via TemporalCurriculumCallback (analogous to SinFusion's
    frame_diff curriculum in FrameSet)

Usage:
    python gen_model/train_conditional.py --data_dir data
"""
import torch
import os
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from gen_model.train_base import default_se3_conf, default_model_conf, default_data_args
from gen_model.se3_diffusion_module import ConditionalSE3Diffusion


class TemporalCurriculumCallback(L.Callback):
    """Increases the temporal-gap curriculum range by 1 every `grow_every` epochs.

    Mirrors SinFusion's FrameSet curriculum: starts at current_max_k=1
    (k ∈ {-1, +1}) and grows to max_k over training.
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


def main():
    parser = argparse.ArgumentParser(description='Conditional SE(3) Diffusion Training')
    parser.add_argument('--data_dir',    type=str,   default='data')
    parser.add_argument('--atlas_csv',   type=str,   default='gen_model/splits/atlas.csv')
    parser.add_argument('--train_split', type=str,   default='gen_model/splits/frame_splits.csv')
    parser.add_argument('--suffix',      type=str,   default='_latent')
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--max_steps',   type=int,   default=200_000)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--save_dir',    type=str,   default='checkpoints/conditional')
    parser.add_argument('--stride',      type=int,   default=1)
    parser.add_argument('--max_k',       type=int,   default=3)
    parser.add_argument('--grow_every',  type=int,   default=10)
    parser.add_argument('--lora_r',      type=int,   default=0)
    parser.add_argument('--lora_alpha',  type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    from gen_model.diffusion.se3_diffuser import SE3Diffuser
    from gen_model.models.score_network import ScoreNetwork
    from gen_model.models.lora import apply_lora
    from gen_model.data.dataset import ConditionalMDGenDataset

    se3_conf   = default_se3_conf()
    model_conf = default_model_conf(
        use_temporal_embedding=True,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    )
    data_args  = default_data_args(args)
    diffuser   = SE3Diffuser(se3_conf)

    # Datasets — no diffuser passed: noise applied in ConditionalSE3Diffusion.forward()
    train_dataset = ConditionalMDGenDataset(
        args=data_args, mode='train',
        stride=args.stride, max_k=args.max_k, current_max_k=1,
    )
    val_dataset = ConditionalMDGenDataset(
        args=data_args, mode='val',
        stride=args.stride, max_k=args.max_k, current_max_k=1,
    )
    val_dataset.coord_scale = float(train_dataset.coord_scale)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    score_network = ScoreNetwork(model_conf, diffuser)
    apply_lora(score_network, model_conf.lora)

    module = ConditionalSE3Diffusion(
        model=score_network,
        diffuser=diffuser,
        lr=args.lr,
        cosine_T_max=args.max_steps,
    )

    curriculum_cb = TemporalCurriculumCallback(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_k=args.max_k,
        grow_every=args.grow_every,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='cond-{step:06d}-{val_loss:.4f}',
        save_top_k=3, monitor='val_loss', mode='min', save_last=True,
    )
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator='auto', devices='auto',
        callbacks=[curriculum_cb, checkpoint_cb],
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
