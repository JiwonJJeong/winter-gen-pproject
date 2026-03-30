"""SE3 diffusion Lightning modules — modelled after SinFusion's Diffusion and ConditionalDiffusion.

Key design (SinFusion pattern):
  forward()         — owns the complete training pipeline:
                       sample t → SE3 forward marginal → model forward → score loss
                       (analogous to SinFusion's q_sample + MSE in one method)
  training_step     — calls forward(), logs, increments step_counter
  validation_step   — same as training_step, no grad
  configure_optimizers — Adam + CosineAnnealingLR (instead of SinFusion's MultiStepLR)

SE3 domain adaptations vs SinFusion:
  - Noise process: SE3Diffuser.forward_marginal (SO3×R3 VP-SDE) replaces DDPM q_sample
  - Loss: scale-normalised score MSE (rot + trans + psi) replaces pixel MSE
  - Conditioning (conditional model): sc_ca_t (source CA positions) replaces CONDITION_IMG
  - Spatial masking: res_mask applied to loss instead of hard crop

EMA (from MDGen): maintains an exponential moving average of model weights
for validation and inference (decay=0.999).
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as L

import gen_model.path_setup  # noqa: F401 — adds extern/se3_diffusion to sys.path
from gen_model.utils.rigid_utils import Rigid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(x, dtype=torch.float32):
    """Convert numpy array or scalar to tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(x, dtype=dtype)


# ---------------------------------------------------------------------------
# EMA (following MDGen's pattern)
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            # Shadow is initialised on CPU; migrate lazily to the model's device.
            # .to() is a no-op when already on the correct device.
            self.shadow[k] = self.shadow[k].to(v.device)
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply(self, model: torch.nn.Module):
        """Swap model weights with EMA weights; returns backup for restore."""
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return backup

    def restore(self, model: torch.nn.Module, backup: dict):
        model.load_state_dict(backup)


# ---------------------------------------------------------------------------
# Unconditional
# ---------------------------------------------------------------------------

class SE3Diffusion(L.LightningModule):
    """Unconditional SE(3) score-matching diffusion.

    Modelled after SinFusion's Diffusion class.  forward() owns t-sampling,
    SE3 forward marginal (analog of q_sample), model forward, and loss.

    Args:
        model:        ScoreNetwork (or StarScoreNetwork).
        diffuser:     SE3Diffuser instance.
        lr:           Initial learning rate.
        min_t:        Minimum diffusion time (avoids t≈0 instability).
        max_t:        Maximum diffusion time.  Set to ~0.1 for Diffusion
                      Forcing (STAR-MD multi-frame); 1.0 for unconditional.
        rot_weight:   Rotation score loss weight (paper: 0.5).
        trans_weight: Translation score loss weight (paper: 1.0).
        psi_weight:   Torsion angle loss weight.
        bb_atom_weight: Backbone atom position loss weight (paper: 1.0, filtered at t < 0.25).
        dist_mat_weight: Distance matrix loss weight (paper: 1.0, filtered at t < 0.25).
        aux_weight:   Multiplier for auxiliary losses (paper: 0.25).
        rot_t_threshold: Only apply rotation loss when t < this (paper: 0.2).
        aux_t_threshold: Only apply aux losses when t < this (paper: 0.25).
        cosine_T_max: CosineAnnealingLR period in steps.
        ema_decay:    EMA decay rate (0 = disabled, 0.999 = MDGen default).
    """

    def __init__(
        self,
        model,
        diffuser,
        lr: float = 1e-4,
        min_t: float = 0.01,
        max_t: float = 1.0,
        rot_weight: float = 0.5,
        trans_weight: float = 1.0,
        psi_weight: float = 1.0,
        bb_atom_weight: float = 1.0,
        dist_mat_weight: float = 1.0,
        aux_weight: float = 0.25,
        rot_t_threshold: float = 0.2,
        aux_t_threshold: float = 0.25,
        cosine_T_max: int = 100_000,
        ema_decay: float = 0.999,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.model = model
        self.diffuser = diffuser
        self.lr = lr
        self.min_t = min_t
        self.max_t = max_t
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight
        self.psi_weight = psi_weight
        self.bb_atom_weight = bb_atom_weight
        self.dist_mat_weight = dist_mat_weight
        self.aux_weight = aux_weight
        self.rot_t_threshold = rot_t_threshold
        self.aux_t_threshold = aux_t_threshold
        self.cosine_T_max = cosine_T_max
        self.warmup_steps = warmup_steps
        self.step_counter = 0  # mirrors SinFusion's step_counter
        self.ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None
        self._ema_backup = None

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # Drop stale inv_freq keys from old checkpoints (now non-persistent).
        checkpoint['state_dict'] = {
            k: v for k, v in checkpoint['state_dict'].items()
            if not k.endswith('.rope.inv_freq')
        }
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']

    # ------------------------------------------------------------------
    # SE3 forward marginal — analog of SinFusion's q_sample
    # ------------------------------------------------------------------

    def _apply_se3_noise(self, batch, t_batch: np.ndarray) -> dict:
        """Apply SE3 forward marginal per sample and inject diffusion features into batch.

        Handles both single-frame and multi-frame batches:
          - Single-frame: rigids_0 [B, N, 7]
          - Multi-frame:  rigids_0 [B, L, N, 7] — same t applied to all L frames
            per batch item (Diffusion Forcing: all frames share one noise level).

        Args:
            batch:   dict with rigids_0 [B, N, 7] or [B, L, N, 7], res_mask [B, N]
            t_batch: [B] numpy array of diffusion times

        Returns:
            batch updated in-place with:
                rigids_t, rot_score, trans_score,
                rot_score_scaling, trans_score_scaling, t
        """
        B = batch['rigids_0'].shape[0]
        multi_frame = batch['rigids_0'].ndim == 4
        device = batch['rigids_0'].device

        if multi_frame:
            L = batch['rigids_0'].shape[1]
            all_rigids_t, all_rot_scores, all_trans_scores = [], [], []
            rot_scalings, trans_scalings = [], []

            for i in range(B):
                t_i = float(t_batch[i])
                mask_i = batch['res_mask'][i].cpu().numpy()
                frame_rigids_t, frame_rot, frame_trans = [], [], []
                for l in range(L):
                    rigids_0_il = Rigid.from_tensor_7(batch['rigids_0'][i, l].float())
                    diff = self.diffuser.forward_marginal(
                        rigids_0=rigids_0_il, t=t_i,
                        diffuse_mask=mask_i, as_tensor_7=True)
                    frame_rigids_t.append(_to_tensor(diff['rigids_t']))
                    frame_rot.append(_to_tensor(diff['rot_score']))
                    frame_trans.append(_to_tensor(diff['trans_score']))
                # Same scaling for all L frames (same t)
                rot_scalings.append(float(diff['rot_score_scaling']))
                trans_scalings.append(float(diff['trans_score_scaling']))
                all_rigids_t.append(torch.stack(frame_rigids_t))    # [L, N, 7]
                all_rot_scores.append(torch.stack(frame_rot))        # [L, N, 3]
                all_trans_scores.append(torch.stack(frame_trans))    # [L, N, 3]

            batch['rigids_t']            = torch.stack(all_rigids_t).to(device)     # [B,L,N,7]
            batch['rot_score']           = torch.stack(all_rot_scores).to(device)   # [B,L,N,3]
            batch['trans_score']         = torch.stack(all_trans_scores).to(device) # [B,L,N,3]
            batch['rot_score_scaling']   = torch.tensor(rot_scalings,  dtype=torch.float32, device=device)
            batch['trans_score_scaling'] = torch.tensor(trans_scalings, dtype=torch.float32, device=device)
            batch['t']                   = torch.tensor(t_batch, dtype=torch.float32, device=device)
        else:
            rigids_t_list, rot_scores, trans_scores = [], [], []
            rot_scalings, trans_scalings = [], []

            for i in range(B):
                rigids_0_i = Rigid.from_tensor_7(batch['rigids_0'][i].float())
                mask_i = batch['res_mask'][i].cpu().numpy()
                t_i = float(t_batch[i])
                diff = self.diffuser.forward_marginal(
                    rigids_0=rigids_0_i, t=t_i, diffuse_mask=mask_i, as_tensor_7=True)
                rigids_t_list.append(_to_tensor(diff['rigids_t']))
                rot_scores.append(_to_tensor(diff['rot_score']))
                trans_scores.append(_to_tensor(diff['trans_score']))
                rot_scalings.append(float(diff['rot_score_scaling']))
                trans_scalings.append(float(diff['trans_score_scaling']))

            batch['rigids_t']            = torch.stack(rigids_t_list).to(device)
            batch['rot_score']           = torch.stack(rot_scores).to(device)
            batch['trans_score']         = torch.stack(trans_scores).to(device)
            batch['rot_score_scaling']   = torch.tensor(rot_scalings,  dtype=torch.float32, device=device)
            batch['trans_score_scaling'] = torch.tensor(trans_scalings, dtype=torch.float32, device=device)
            batch['t']                   = torch.tensor(t_batch, dtype=torch.float32, device=device)

        return batch

    # ------------------------------------------------------------------
    # Loss — scale-normalised score MSE (analog of SinFusion's pixel MSE)
    # ------------------------------------------------------------------

    def _score_loss(self, pred: dict, batch: dict, mask: torch.Tensor):
        """Scale-normalised MSE over rotation score, translation score, psi,
        and auxiliary backbone/distance losses.

        Following STAR-MD paper appendix:
          - Rotation loss filtered by t < rot_t_threshold (default 0.2)
          - Backbone atom + distance matrix auxiliary losses at t < aux_t_threshold (0.25)
          - Auxiliary losses scaled by aux_weight (0.25)

        Handles both single-frame [B, N, 3] and multi-frame [B, L, N, 3] scores.
        For multi-frame batches, mask [B, N] is expanded to [B, L, N] so all L
        frames contribute equally to the loss (Diffusion Forcing).

        Returns:
            (total, rot_loss, trans_loss, psi_loss)
        """
        multi_frame = pred['rot_score'].ndim == 4  # [B, L, N, 3]
        t = batch['t'].float()  # [B]

        if multi_frame:
            L = pred['rot_score'].shape[1]
            rot_scaling   = batch['rot_score_scaling'].float()[:, None, None, None] + 1e-8
            trans_scaling = batch['trans_score_scaling'].float()[:, None, None, None] + 1e-8
            # Expand res_mask [B, N] → [B, L, N]
            mask_exp = mask[:, None, :].expand(-1, L, -1)
        else:
            rot_scaling   = batch['rot_score_scaling'].float()[:, None, None] + 1e-8
            trans_scaling = batch['trans_score_scaling'].float()[:, None, None] + 1e-8
            mask_exp = mask

        # ── Core score losses ─────────────────────────────────────────────

        rot_mse = F.mse_loss(
            pred['rot_score']          / rot_scaling,
            batch['rot_score'].float() / rot_scaling,
            reduction='none',
        ).sum(dim=-1)   # [B, N] or [B, L, N]

        trans_mse = F.mse_loss(
            pred['trans_score']          / trans_scaling,
            batch['trans_score'].float() / trans_scaling,
            reduction='none',
        ).sum(dim=-1)   # [B, N] or [B, L, N]

        n_visible  = mask_exp.sum() + 1e-8

        # Time-filtered rotation loss (paper: only apply when t < 0.2)
        rot_t_mask = (t < self.rot_t_threshold).float()  # [B]
        if multi_frame:
            rot_t_mask = rot_t_mask[:, None, None].expand_as(rot_mse)
        else:
            rot_t_mask = rot_t_mask[:, None].expand_as(rot_mse)
        # Normalise by the tokens that actually contribute (t < threshold),
        # not all visible tokens — otherwise rot_loss is underweighted whenever
        # some batch items have t >= rot_t_threshold (their contribution is 0).
        n_rot = (mask_exp * rot_t_mask).sum() + 1e-8
        rot_loss = (rot_mse * mask_exp * rot_t_mask).sum() / n_rot

        trans_loss = (trans_mse * mask_exp).sum() / n_visible

        gt_psi  = batch['torsion_angles_sin_cos'][..., 2, :].float()
        psi_mse = F.mse_loss(pred['psi'], gt_psi, reduction='none').sum(dim=-1)
        psi_loss = (psi_mse * mask_exp).sum() / n_visible

        total = (self.rot_weight   * rot_loss
               + self.trans_weight * trans_loss
               + self.psi_weight   * psi_loss)

        # ── Auxiliary losses (paper: filtered at t < 0.25, weight 0.25) ──

        aux_t_mask = (t < self.aux_t_threshold).float()  # [B]
        has_aux = aux_t_mask.sum() > 0 and self.aux_weight > 0

        if has_aux and 'atom37' in pred and 'atom37_pos' in batch:
            # Backbone atom position loss
            pred_bb = pred['atom37']        # [B, N, 37, 3] (single-frame only)
            gt_bb   = batch['atom37_pos'].float()
            bb_mask = (gt_bb.abs().sum(-1) > 1e-6).float()  # [B, N, 37]
            bb_mse  = ((pred_bb - gt_bb) ** 2).sum(-1)      # [B, N, 37]
            bb_loss = (bb_mse * bb_mask * mask[..., None]).sum() / (bb_mask.sum() + 1e-8)
            bb_loss = bb_loss * aux_t_mask.mean()  # scale by fraction of samples below threshold
            total = total + self.aux_weight * self.bb_atom_weight * bb_loss

            # Distance matrix loss (pairwise CA-CA)
            pred_ca = pred_bb[:, :, 1, :]  # [B, N, 3] — CA is index 1 in atom37
            gt_ca   = gt_bb[:, :, 1, :]
            # Squared distances directly (avoids sqrt gradient NaN at zero).
            def _sq_dist(x):
                d = x.unsqueeze(2) - x.unsqueeze(1)   # [B, N, N, 3]
                return d.pow(2).sum(-1)                # [B, N, N]
            pred_dist = _sq_dist(pred_ca)
            gt_dist   = _sq_dist(gt_ca)
            dist_mask = mask[:, :, None] * mask[:, None, :]  # [B, N, N]
            dist_mse  = ((pred_dist - gt_dist) ** 2 * dist_mask).sum() / (dist_mask.sum() + 1e-8)
            dist_mse  = dist_mse * aux_t_mask.mean()
            total = total + self.aux_weight * self.dist_mat_weight * dist_mse

        return total, rot_loss, trans_loss, psi_loss

    # ------------------------------------------------------------------
    # forward — owns the complete training pipeline (SinFusion pattern)
    # ------------------------------------------------------------------

    def forward(self, batch: dict):
        """Sample t → SE3 forward marginal → model → score loss.

        SinFusion analogy:
            t        = randint(0, T)
            α̂       = uniform(√α̂_{t-1}, √α̂_t)    ← continuous noise level
            x_noisy  = q_sample(x_clean, α̂)         ← _apply_se3_noise
            pred     = model(x_noisy, α̂)
            loss     = MSE(noise, pred)              ← _score_loss

        SE3 diffusion is already continuous in t (unlike DDPM's discrete steps),
        so we use stratified sampling: divide [min_t, max_t] into B equal strata,
        sample one t per stratum.  This ensures each batch covers the full noise
        range without gaps — the SE3 analog of SinFusion's continuous α̂ sampling.
        """
        B = batch['rigids_0'].shape[0]

        # 1. Stratified t sampling (SinFusion-style continuous noise coverage).
        #    Divide [min_t, max_t] into B strata; sample one t per stratum.
        #    With B=1 this reduces to uniform, but with B>1 it guarantees
        #    every batch covers the full noise range.
        strata = np.linspace(self.min_t, self.max_t, B + 1)
        t_batch = np.array([
            np.random.uniform(strata[i], strata[i + 1]) for i in range(B)
        ])

        # 2. SE3 forward marginal (SinFusion: x_noisy = q_sample(x_clean, t))
        batch = self._apply_se3_noise(batch, t_batch)

        # 3. Model forward (SinFusion: pred = model(x_noisy, t))
        pred = self.model(batch)

        # 4. Score loss (SinFusion: MSE(noise, pred))
        mask = batch['res_mask'].float()
        return self._score_loss(pred, batch, mask)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def _log_losses(self, split: str, loss, rot, trans, psi, bs: int):
        on_step  = (split == 'train')
        on_epoch = (split != 'train')   # epoch agg meaningless for virtual-epoch step-based training
        self.log(f'{split}_loss',       loss,  on_step=on_step, on_epoch=on_epoch, prog_bar=True, batch_size=bs)
        self.log(f'{split}_rot_loss',   rot,   on_step=on_step, on_epoch=on_epoch,               batch_size=bs)
        self.log(f'{split}_trans_loss', trans, on_step=on_step, on_epoch=on_epoch,               batch_size=bs)
        self.log(f'{split}_psi_loss',   psi,   on_step=on_step, on_epoch=on_epoch,               batch_size=bs)

    def training_step(self, batch, batch_idx):
        loss, rot, trans, psi = self.forward(batch)
        self._log_losses('train', loss, rot, trans, psi, batch['res_mask'].shape[0])
        self.step_counter += 1
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after the optimizer step (must use on_train_batch_end,
        not on_before_optimizer_step, so shadow weights see w_{t+1} not w_t)."""
        if self.ema is not None:
            self.ema.update(self.model)

    def on_validation_epoch_start(self):
        """Swap to EMA weights for validation (MDGen pattern)."""
        if self.ema is not None:
            self._ema_backup = self.ema.apply(self.model)

    def on_validation_epoch_end(self):
        """Restore training weights after validation."""
        if self.ema is not None and self._ema_backup is not None:
            self.ema.restore(self.model, self._ema_backup)
            self._ema_backup = None

    def validation_step(self, batch, batch_idx):
        loss, rot, trans, psi = self.forward(batch)
        self._log_losses('val', loss, rot, trans, psi, batch['res_mask'].shape[0])
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.cosine_T_max, eta_min=self.lr * 0.01)
        if self.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optim, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_steps)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optim, schedulers=[warmup, cosine], milestones=[self.warmup_steps])
        else:
            scheduler = cosine
        return {
            'optimizer': optim,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }


# ---------------------------------------------------------------------------
# Conditional
# ---------------------------------------------------------------------------

class ConditionalSE3Diffusion(SE3Diffusion):
    """Conditional SE(3) diffusion — modelled after SinFusion's ConditionalDiffusion.

    The source frame's CA positions (sc_ca_t) condition the model,
    analogous to SinFusion's CONDITION_IMG.  The dataset
    (ConditionalMDGenDataset) already places sc_ca_t and k in the batch;
    forward() only needs to noise the TARGET rigids_0 and run the model.

    SinFusion analogy:
        CONDITION_IMG  ↔  sc_ca_t  (source frame CA, centred + scaled)
        IMG            ↔  rigids_0 (target frame clean structure)
        FRAME          ↔  k        (signed temporal stride)
    """

    # forward() is inherited unchanged — sc_ca_t and k are already in batch,
    # ScoreNetwork reads them internally (temporal embedding uses k via Embedder).
    # No override needed: the SinFusion pattern works as-is.
