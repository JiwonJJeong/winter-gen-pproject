"""STAR-MD conditional SE(3) diffusion training (Diffusion Forcing).

Training scheme:
  - ConditionalMDGenDataset returns L-frame windows; delta_t is sampled from
    LogUniform[0.01, 10] ns and the frame stride k is derived from it.
  - All L frames are noised at the same tau ~ U[min_t, max_t] inside
    ConditionalSE3Diffusion.forward() (SinFusion pattern).
  - Loss is computed at all L positions simultaneously (Diffusion Forcing).
  - Block-causal SpatioTemporalAttention provides the causal structure.

References:
  - SinFusion: single-trajectory DDPM with step-based training, curriculum learning
  - MDGen: gradient clipping, EMA, num_workers, accumulate_grad_batches

Usage:
    python gen_model/train_conditional.py --data_dir data
"""
import torch
import os
import argparse
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from gen_model.train_base import default_se3_conf, default_model_conf, default_data_args
from gen_model.se3_diffusion_module import ConditionalSE3Diffusion


def main():
    parser = argparse.ArgumentParser(description='STAR-MD Conditional SE(3) Diffusion Training')
    parser.add_argument('--data_dir',            type=str,   default='data')
    parser.add_argument('--atlas_csv',           type=str,   default='gen_model/splits/atlas.csv')
    parser.add_argument('--suffix',              type=str,   default='_latent')
    parser.add_argument('--protein',              type=str,   required=True,
                        help='Protein name to train on (single trajectory, SinFusion-style)')
    parser.add_argument('--replica',              type=str,   default='1',
                        help='Replica suffix (e.g. "1" for _R1); single-trajectory training uses one replica')
    parser.add_argument('--batch_size',          type=int,   default=1,
                        help='Batch size (SinFusion default: 1 for single-trajectory training)')
    parser.add_argument('--max_steps',           type=int,   default=200_000)
    parser.add_argument('--lr',                  type=float, default=1e-4)
    parser.add_argument('--save_dir',            type=str,   default='checkpoints/conditional')
    parser.add_argument('--stride',              type=int,   default=1)
    parser.add_argument('--num_frames',          type=int,   default=8,
                        help='Window length L (paper training: 8; inference: 80)')
    parser.add_argument('--ns_per_stored_frame', type=float, default=0.1,
                        help='Physical time between consecutive stored frames in ns')
    parser.add_argument('--min_t',               type=float, default=0.01,
                        help='Minimum diffusion time')
    parser.add_argument('--max_t',               type=float, default=0.1,
                        help='Maximum diffusion time for Diffusion Forcing (narrow noise)')
    parser.add_argument('--lora_r',              type=int,   default=0)
    parser.add_argument('--lora_alpha',          type=float, default=0.0)
    parser.add_argument('--star_enabled',        action='store_true', default=True,
                        help='Enable STAR-MD spatio-temporal attention (default: on)')
    parser.add_argument('--no_star',             dest='star_enabled', action='store_false',
                        help='Disable STAR-MD spatio-temporal attention')
    parser.add_argument('--st_num_heads',        type=int,   default=8,
                        help='ST attention heads (paper: 8)')
    parser.add_argument('--spatial_sigma',       type=float, default=0.0,
                        help='Spatial Gaussian sigma for ST attention (Angstroms). '
                             '0 = global. SinFusion-inspired local receptive field.')
    parser.add_argument('--num_blocks',          type=int,   default=8,
                        help='Number of IPA blocks (paper: 8; Colab: 4 for speed)')
    # MDGen-inspired options
    parser.add_argument('--grad_clip',           type=float, default=1.0,
                        help='Gradient clipping value (MDGen default: 1.0)')
    parser.add_argument('--ema_decay',           type=float, default=0.999,
                        help='EMA decay rate (0 = disabled, MDGen default: 0.999)')
    parser.add_argument('--accumulate_grad',     type=int,   default=1,
                        help='Accumulate gradients over N batches (MDGen-style)')
    parser.add_argument('--num_workers',         type=int,   default=4,
                        help='DataLoader workers')
    parser.add_argument('--virtual_epoch_size',  type=int,   default=5000,
                        help='Virtual epoch size (SinFusion default: 5000; 0 = use real dataset size)')
    # SinFusion curriculum learning for delta_t
    parser.add_argument('--curriculum',          action='store_true', default=True,
                        help='Enable SinFusion-style curriculum for delta_t range')
    parser.add_argument('--no_curriculum',       dest='curriculum', action='store_false',
                        help='Disable curriculum learning')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Resume from checkpoint (restores weights, optimizer, step counter)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    from gen_model.diffusion.se3_diffuser import SE3Diffuser
    from gen_model.models.star_score_network import StarScoreNetwork
    from gen_model.models.lora import apply_lora
    from gen_model.data.dataset import ConditionalMDGenDataset

    se3_conf   = default_se3_conf()
    model_conf = default_model_conf(
        use_temporal_embedding=False,   # delta_t handled by AdaLN cond, not Embedder
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        star_enabled=args.star_enabled,
        st_num_heads=args.st_num_heads,
        spatial_sigma=args.spatial_sigma,
        num_blocks=args.num_blocks,
    )
    data_args  = default_data_args(args)
    diffuser   = SE3Diffuser(se3_conf)

    # Dataset ��� use all frames (SinFusion-style: no train/val split within one trajectory).
    # Evaluation is done post-hoc against held-out replicas via evaluate.py.
    train_dataset = ConditionalMDGenDataset(
        args=data_args, mode='all',
        stride=args.stride,
        num_frames=args.num_frames,
        ns_per_stored_frame=args.ns_per_stored_frame,
        curriculum=args.curriculum,
        virtual_epoch_size=args.virtual_epoch_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    score_network = StarScoreNetwork(model_conf, diffuser)
    apply_lora(score_network, model_conf.lora)

    module = ConditionalSE3Diffusion(
        model=score_network,
        diffuser=diffuser,
        lr=args.lr,
        min_t=args.min_t,
        max_t=args.max_t,
        cosine_T_max=args.max_steps,
        ema_decay=args.ema_decay,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='cond-{step:06d}-{train_loss:.4f}',
        save_top_k=3, monitor='train_loss', mode='min', save_last=True,
    )
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator='auto', devices='auto',
        callbacks=[checkpoint_cb],
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=500,
        enable_progress_bar=False,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad,
    )
    trainer.fit(module, train_dataloaders=train_loader, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
