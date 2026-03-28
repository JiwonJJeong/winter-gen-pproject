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
"""

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

import gen_model.path_setup  # noqa: F401 — adds extern/se3_diffusion to sys.path
from openfold.utils.rigid_utils import Rigid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(x, dtype=torch.float32):
    """Convert numpy array or scalar to tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(x, dtype=dtype)


# ---------------------------------------------------------------------------
# Unconditional
# ---------------------------------------------------------------------------

class SE3Diffusion(L.LightningModule):
    """Unconditional SE(3) score-matching diffusion.

    Modelled after SinFusion's Diffusion class.  forward() owns t-sampling,
    SE3 forward marginal (analog of q_sample), model forward, and loss.

    Args:
        model:        ScoreNetwork (or StarIpaScore wrapper).
        diffuser:     SE3Diffuser instance.
        lr:           Initial learning rate.
        min_t:        Minimum diffusion time (avoids t≈0 instability).
        rot_weight:   Rotation score loss weight.
        trans_weight: Translation score loss weight.
        psi_weight:   Torsion angle loss weight.
        cosine_T_max: CosineAnnealingLR period in steps.
    """

    def __init__(
        self,
        model,
        diffuser,
        lr: float = 1e-4,
        min_t: float = 0.01,
        rot_weight: float = 1.0,
        trans_weight: float = 1.0,
        psi_weight: float = 1.0,
        cosine_T_max: int = 100_000,
    ):
        super().__init__()
        self.model = model
        self.diffuser = diffuser
        self.lr = lr
        self.min_t = min_t
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight
        self.psi_weight = psi_weight
        self.cosine_T_max = cosine_T_max
        self.step_counter = 0  # mirrors SinFusion's step_counter

    # ------------------------------------------------------------------
    # SE3 forward marginal — analog of SinFusion's q_sample
    # ------------------------------------------------------------------

    def _apply_se3_noise(self, batch, t_batch: np.ndarray) -> dict:
        """Apply SE3 forward marginal per sample and inject diffusion features into batch.

        Args:
            batch:   dict with rigids_0 [B, N, 7], res_mask [B, N]
            t_batch: [B] numpy array of diffusion times in [min_t, 1.0]

        Returns:
            batch updated in-place with:
                rigids_t, rot_score, trans_score,
                rot_score_scaling, trans_score_scaling, t
        """
        B = batch['rigids_0'].shape[0]
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

        device = batch['rigids_0'].device
        batch['rigids_t']           = torch.stack(rigids_t_list).to(device)
        batch['rot_score']          = torch.stack(rot_scores).to(device)
        batch['trans_score']        = torch.stack(trans_scores).to(device)
        batch['rot_score_scaling']  = torch.tensor(rot_scalings,  dtype=torch.float32, device=device)
        batch['trans_score_scaling']= torch.tensor(trans_scalings, dtype=torch.float32, device=device)
        batch['t']                  = torch.tensor(t_batch, dtype=torch.float32, device=device)
        return batch

    # ------------------------------------------------------------------
    # Loss — scale-normalised score MSE (analog of SinFusion's pixel MSE)
    # ------------------------------------------------------------------

    def _score_loss(self, pred: dict, batch: dict, mask: torch.Tensor):
        """Scale-normalised MSE over rotation score, translation score, and psi.

        Returns:
            (total, rot_loss, trans_loss, psi_loss)
        """
        rot_scaling   = batch['rot_score_scaling'].float()[:, None, None]   + 1e-8
        trans_scaling = batch['trans_score_scaling'].float()[:, None, None] + 1e-8

        rot_mse = F.mse_loss(
            pred['rot_score']   / rot_scaling,
            batch['rot_score'].float()   / rot_scaling,
            reduction='none',
        ).sum(dim=-1)   # [B, N]

        trans_mse = F.mse_loss(
            pred['trans_score'] / trans_scaling,
            batch['trans_score'].float() / trans_scaling,
            reduction='none',
        ).sum(dim=-1)   # [B, N]

        n_visible = mask.sum() + 1e-8
        rot_loss   = (rot_mse   * mask).sum() / n_visible
        trans_loss = (trans_mse * mask).sum() / n_visible

        gt_psi   = batch['torsion_angles_sin_cos'][..., 2, :].float()  # [B, N, 2]
        psi_mse  = F.mse_loss(pred['psi'], gt_psi, reduction='none').sum(dim=-1)
        psi_loss = (psi_mse * mask).sum() / n_visible

        total = (self.rot_weight   * rot_loss
               + self.trans_weight * trans_loss
               + self.psi_weight   * psi_loss)
        return total, rot_loss, trans_loss, psi_loss

    # ------------------------------------------------------------------
    # forward — owns the complete training pipeline (SinFusion pattern)
    # ------------------------------------------------------------------

    def forward(self, batch: dict):
        """Sample t → SE3 forward marginal → model → score loss.

        SinFusion analogy:
            t        = randint(0, T)
            x_noisy  = q_sample(x_clean, t)      ← _apply_se3_noise
            pred     = model(x_noisy, t)
            loss     = MSE(noise, pred)           ← _score_loss
        """
        B = batch['rigids_0'].shape[0]

        # 1. Sample t per sample (SinFusion: t = randint(0, T))
        t_batch = np.random.uniform(self.min_t, 1.0, size=B)

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
        on_step = (split == 'train')
        self.log(f'{split}_loss',       loss,  on_step=on_step, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f'{split}_rot_loss',   rot,   on_step=False,   on_epoch=True,                batch_size=bs)
        self.log(f'{split}_trans_loss', trans, on_step=False,   on_epoch=True,                batch_size=bs)
        self.log(f'{split}_psi_loss',   psi,   on_step=False,   on_epoch=True,                batch_size=bs)

    def training_step(self, batch, batch_idx):
        loss, rot, trans, psi = self.forward(batch)
        self._log_losses('train', loss, rot, trans, psi, batch['res_mask'].shape[0])
        self.step_counter += 1
        return loss

    def validation_step(self, batch, batch_idx):
        loss, rot, trans, psi = self.forward(batch)
        self._log_losses('val', loss, rot, trans, psi, batch['res_mask'].shape[0])
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.cosine_T_max, eta_min=self.lr * 0.01)
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
