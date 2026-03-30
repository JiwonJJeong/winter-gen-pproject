"""MDGen architecture + SinFusion anti-overfitting techniques.

Uses MDGen's own model (LatentMDGenModel + flow matching transport) but
applies SinFusion-inspired training tricks that MDGen's CLI doesn't support:

  1. Spatial cropping via 3D-distance mask on the loss (crop_ratio=0.95)
  2. Stratified noise sampling (uniform t coverage per batch)
  3. Virtual epoch size (5000 samples/epoch)

This isolates whether improvements come from:
  - STAR-MD architecture (joint ST attention, 2D-RoPE, AdaLN)
  - SinFusion training protocol (the tricks above)

Usage:
    python gen_model/train_mdgen_enhanced.py \\
        --protein 4o66_C --replica 1 --data_dir data
"""

import argparse
import os
import sys

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from torch.utils.data import DataLoader, Dataset

import gen_model.path_setup  # noqa: F401

from mdgen.dataset import MDGenDataset
from mdgen.wrapper import NewMDGenWrapper
from mdgen.parsing import parse_train_args


# ---------------------------------------------------------------------------
# Spatial cropping (SinFusion-style, applied to MDGen's loss mask)
# ---------------------------------------------------------------------------

def spatial_crop_mask(ca_pos: torch.Tensor, mask: torch.Tensor,
                      crop_ratio: float = 0.95) -> torch.Tensor:
    """Build a spatial crop mask based on 3D CA-CA distance.

    Randomly selects a seed residue and keeps the nearest crop_ratio fraction.

    Args:
        ca_pos:     [B, T, L, 3] or [B, L, 3] CA positions
        mask:       [B, T, L] or [B, L] existing validity mask
        crop_ratio: Fraction of residues to keep

    Returns:
        crop_mask: same shape as mask, with 0 for cropped residues
    """
    if ca_pos.ndim == 4:
        # Use first frame for distance computation
        ca = ca_pos[:, 0]   # [B, L, 3]
        m = mask[:, 0]      # [B, L]
    else:
        ca = ca_pos
        m = mask

    B, L, _ = ca.shape

    crop_mask = torch.ones_like(m)
    for b in range(B):
        valid = m[b].bool()
        n_valid = valid.sum().item()
        # k is a fraction of *valid* residues, not the padded length.
        # Using L here would make k > n_valid for most padded sequences,
        # causing the crop to be silently skipped every time.
        k = max(1, int(n_valid * crop_ratio))
        if n_valid <= k:
            continue
        # Random seed from valid residues
        valid_idx = torch.where(valid)[0]
        seed = valid_idx[torch.randint(len(valid_idx), (1,))].item()
        # Nearest k by CA distance
        dists = torch.norm(ca[b] - ca[b, seed:seed+1], dim=-1)
        dists[~valid] = float('inf')
        _, keep = torch.topk(dists, k=k, largest=False)
        new_mask = torch.zeros(L, device=m.device)
        new_mask[keep] = 1.0
        crop_mask[b] = new_mask

    # Expand to match mask shape
    if mask.ndim == 3:
        crop_mask = crop_mask.unsqueeze(1).expand_as(mask)

    return crop_mask


# ---------------------------------------------------------------------------
# Stratified noise override
# ---------------------------------------------------------------------------

class StratifiedTransportWrapper:
    """Wraps MDGen's Transport to use stratified t sampling."""

    def __init__(self, transport):
        self._transport = transport
        # Copy all attributes
        for attr in dir(transport):
            if not attr.startswith('_') and attr not in ('sample', 'training_losses'):
                try:
                    setattr(self, attr, getattr(transport, attr))
                except (AttributeError, TypeError):
                    pass

    def sample(self, x1):
        """Stratified t sampling: divide [t0, t1] into B strata."""
        x0 = torch.randn_like(x1)
        t0, t1 = self._transport.check_interval(
            self._transport.train_eps, self._transport.sample_eps)
        B = x1.shape[0]
        strata = np.linspace(t0, t1, B + 1)
        t_np = np.array([np.random.uniform(strata[i], strata[i+1]) for i in range(B)])
        t = torch.tensor(t_np, dtype=x1.dtype, device=x1.device)
        return t, x0, x1

    def training_losses(self, **kwargs):
        return self._transport.training_losses(**kwargs)

    def __getattr__(self, name):
        return getattr(self._transport, name)


# ---------------------------------------------------------------------------
# Virtual epoch dataset wrapper
# ---------------------------------------------------------------------------

class VirtualEpochDataset(Dataset):
    """Wraps a dataset to report a virtual size and sample randomly."""

    def __init__(self, dataset, virtual_size: int = 5000):
        self.dataset = dataset
        self.virtual_size = virtual_size

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, idx):
        real_idx = np.random.randint(len(self.dataset))
        return self.dataset[real_idx]


# ---------------------------------------------------------------------------
# Enhanced wrapper (spatial crop + stratified noise)
# ---------------------------------------------------------------------------

class EnhancedMDGenWrapper(NewMDGenWrapper):
    """NewMDGenWrapper with SinFusion anti-overfitting applied.

    Additions over NewMDGenWrapper:
      - Spatial cropping on loss mask (crop_ratio=0.95)
      - Stratified noise sampling
      - Gaussian spatial attention bias on mha_l (spatial_sigma)
    """

    def __init__(self, args):
        super().__init__(args)
        self.crop_ratio = getattr(args, 'crop_ratio', 0.95)
        self.spatial_sigma = getattr(args, 'spatial_sigma', 15.0)
        # Replace transport with stratified version
        self.transport = StratifiedTransportWrapper(self.transport)
        # Monkey-patch spatial attention with Gaussian distance bias
        if self.spatial_sigma > 0:
            self._patch_spatial_attention()

    def _patch_spatial_attention(self):
        """Inject Gaussian distance bias into MDGen's mha_l (spatial attention).

        MDGen's LatentMDGenLayer.forward() calls mha_l with (x, mask) — no way
        to pass CA positions without modifying extern.  Instead, we store CA
        positions on each layer object before the forward pass, and the patched
        mha_l.forward reads them from the layer.

        The patched forward computes CA-CA distance → Gaussian bias → additive
        attn_mask in MultiheadAttention.
        """
        sigma = self.spatial_sigma

        for layer in self.model.layers:
            if not hasattr(layer, 'mha_l'):
                continue
            layer._spatial_sigma = sigma
            layer._ca_pos = None  # set before each forward pass

            original_forward = layer.mha_l.forward

            def make_patched_forward(orig_fwd, lay):
                def patched_forward(x, mask):
                    # x: [B*T, L, C], mask: [B*T, L]
                    ca = lay._ca_pos
                    if ca is not None and lay._spatial_sigma > 0:
                        # ca: [B, T, L, 3] → [B*T, L, 3]
                        B, T, L_res = ca.shape[:3]
                        ca_flat = ca.reshape(B * T, L_res, 3)
                        # Squared distances directly — avoids sqrt whose gradient
                        # is undefined at zero (diagonal), which would NaN-poison
                        # the backward pass via cdist.
                        diff = ca_flat.unsqueeze(2) - ca_flat.unsqueeze(1)  # [B*T, L, L, 3]
                        dist_sq = diff.pow(2).sum(-1)                        # [B*T, L, L]
                        bias = -(dist_sq / lay._spatial_sigma ** 2)
                        num_heads = lay.mha_heads
                        # [B*T, L, L] → [B*T*H, L, L]
                        bias = bias.unsqueeze(1).expand(-1, num_heads, -1, -1)
                        bias = bias.reshape(B * T * num_heads, L_res, L_res)

                        x_t = x.transpose(0, 1)
                        x_t, _ = lay.mha_l.attn(
                            query=x_t, key=x_t, value=x_t,
                            key_padding_mask=1 - mask,
                            attn_mask=bias)
                        return x_t.transpose(0, 1)
                    else:
                        return orig_fwd(x, mask)
                return patched_forward

            layer.mha_l.forward = make_patched_forward(original_forward, layer)
        print(f'Patched {len(self.model.layers)} mha_l layers with spatial_sigma={sigma}')

    def general_step(self, batch, stage='train'):
        self.iter_step += 1
        self.stage = stage
        import time
        start1 = time.time()

        prep = self.prep_batch(batch)

        # Apply spatial cropping to loss mask (SinFusion-style)
        if stage == 'train' and self.crop_ratio < 1.0:
            # Get CA positions from rigids (translation component)
            if 'rigids' in prep:
                ca = prep['rigids'].get_trans()  # [B, T, L, 3]
                mask_for_crop = prep['model_kwargs']['mask']  # [B, T, L]
                crop = spatial_crop_mask(ca, mask_for_crop, self.crop_ratio)
                # Apply crop to loss_mask: zero out cropped residues
                prep['loss_mask'] = prep['loss_mask'] * crop.unsqueeze(-1)

        # Set CA positions on each layer for Gaussian spatial attention bias.
        # Always update (including to None) so stale positions from the previous
        # batch aren't used when rigids are absent.
        if self.spatial_sigma > 0:
            ca = prep['rigids'].get_trans() if 'rigids' in prep else None
            for layer in self.model.layers:
                if hasattr(layer, '_ca_pos'):
                    layer._ca_pos = ca

        start = time.time()
        out_dict = self.transport.training_losses(
            model=self.model,
            x1=prep['latents'],
            aatype1=batch['seqres'] if self.args.design else None,
            mask=prep['loss_mask'],
            model_kwargs=prep['model_kwargs']
        )
        self.log('model_dur', time.time() - start)
        loss = out_dict['loss']
        self.log('loss', loss)
        self.log('time', out_dict['t'])
        self.log('dur', time.time() - self.last_log_time)
        if 'name' in batch:
            self.log('name', ','.join(batch['name']))
        self.log('general_step_dur', time.time() - start1)
        self.last_log_time = time.time()
        return loss.mean()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # Parse MDGen's standard args, then add our extras
    parser = argparse.ArgumentParser(
        description='MDGen + SinFusion anti-overfitting tricks')
    parser.add_argument('--protein',     type=str, required=True)
    parser.add_argument('--replica',     type=str, default='1')
    parser.add_argument('--data_dir',    type=str, default='data')
    parser.add_argument('--atlas_csv',   type=str, default='gen_model/splits/atlas.csv')
    parser.add_argument('--suffix',      type=str, default='_latent')
    parser.add_argument('--save_dir',    type=str, default='checkpoints/mdgen_enhanced')
    parser.add_argument('--epochs',      type=int, default=500)
    parser.add_argument('--crop_ratio',  type=float, default=0.95,
                        help='Spatial crop ratio (SinFusion: 0.95)')
    parser.add_argument('--spatial_sigma', type=float, default=15.0,
                        help='Gaussian spatial bias sigma for mha_l (0=disabled)')
    parser.add_argument('--virtual_epoch_size', type=int, default=5000)
    parser.add_argument('--num_frames',  type=int, default=250)
    our_args = parser.parse_args()

    save_dir = os.path.join(our_args.save_dir, our_args.protein)
    os.makedirs(save_dir, exist_ok=True)
    os.environ['MODEL_DIR'] = save_dir

    # Create single-protein split CSV
    import pandas as pd
    atlas_df = pd.read_csv(our_args.atlas_csv, index_col='name')
    split_path = os.path.join(save_dir, 'single_protein_split.csv')
    atlas_df.loc[[our_args.protein]].to_csv(split_path)

    # Build MDGen args namespace (simulating parse_train_args output)
    from types import SimpleNamespace
    args = SimpleNamespace(
        # Trainer
        ckpt=None, validate=False, num_workers=4,
        # Epoch
        epochs=our_args.epochs, overfit=False, overfit_peptide=our_args.protein,
        overfit_frame=False,
        train_batches=None, val_batches=None, val_repeat=1,
        inference_batches=0, batch_size=1,
        val_freq=None, val_epoch_freq=1, no_validate=True,
        designability_freq=1,
        # Logging
        print_freq=100, ckpt_freq=50, wandb=False,
        run_name=f'mdgen_enhanced_{our_args.protein}',
        # Optimization
        accumulate_grad=1, grad_clip=1.0, check_grad=False,
        grad_checkpointing=False, adamW=False,
        ema=True, ema_decay=0.999,
        lr=1e-4, precision='32-true',
        # Data
        train_split=split_path, val_split=split_path,
        data_dir=our_args.data_dir,
        num_frames=our_args.num_frames,
        crop=256,
        suffix=f'_R{our_args.replica}{our_args.suffix}',
        atlas=True, copy_frames=False, no_pad=False, short_md=False,
        # Masking
        design_key_frames=False, no_aa_emb=False,
        no_torsion=False, no_design_torsion=False,
        supervise_no_torsions=False, supervise_all_torsions=False,
        # Ablations
        no_offsets=False, no_frames=False,
        # Model
        hyena=False, no_rope=False, dropout=0.0, scale_factor=1.0,
        interleave_ipa=False, prepend_ipa=True, oracle=False,
        num_layers=5, embed_dim=384, mha_heads=16,
        ipa_heads=4, ipa_head_dim=32, ipa_qk=8, ipa_v=8,
        time_multiplier=100.0, abs_pos_emb=False, abs_time_emb=False,
        # Transport
        path_type='GVP', prediction='velocity', sampling_method='dopri5',
        alpha_max=8, discrete_loss_weight=0.5, dirichlet_flow_temp=1.0,
        allow_nan_cfactor=False,
        # Video
        tps_condition=False, design=False, design_from_traj=False,
        sim_condition=True, inpainting=False,
        dynamic_mpnn=False, mpnn=False,
        frame_interval=None, cond_interval=None,
        # Our extras
        crop_ratio=our_args.crop_ratio,
        spatial_sigma=our_args.spatial_sigma,
    )

    # Build datasets
    trainset = MDGenDataset(args, split=args.train_split)
    if our_args.virtual_epoch_size > 0:
        trainset = VirtualEpochDataset(trainset, our_args.virtual_epoch_size)

    train_loader = DataLoader(
        trainset, batch_size=1, num_workers=4,
        shuffle=True, pin_memory=True, persistent_workers=True,
    )

    # Build enhanced model
    model = EnhancedMDGenWrapper(args)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'auto',
        max_epochs=args.epochs,
        limit_train_batches=1.0,
        limit_val_batches=0.0,
        num_sanity_val_steps=0,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        default_root_dir=save_dir,
        callbacks=[
            ModelCheckpoint(
                dirpath=save_dir,
                save_top_k=-1,
                every_n_epochs=args.ckpt_freq,
            ),
            ModelSummary(max_depth=2),
        ],
        accumulate_grad_batches=args.accumulate_grad,
        logger=False,
    )

    print(f'Training MDGen enhanced on {our_args.protein}_R{our_args.replica}')
    print(f'  SinFusion tricks: crop_ratio={our_args.crop_ratio}, '
          f'spatial_sigma={our_args.spatial_sigma}, '
          f'virtual_epoch={our_args.virtual_epoch_size}, stratified_t=True, ema=True')
    print(f'  Save dir: {save_dir}')

    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
