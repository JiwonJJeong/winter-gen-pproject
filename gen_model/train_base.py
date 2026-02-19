"""Shared base Lightning module for SE(3) score-matching diffusion.

Both unconditional and conditional DDPM inherit from SE3BaseModule.
Subclasses only need to set self.model and optionally override _prepare_batch.
"""
import torch
import torch.nn.functional as F
import lightning as L


class SE3BaseModule(L.LightningModule):
    """Abstract base for SE(3) score-matching Lightning modules.

    Subclasses must assign self.model (a ScoreNetwork or compatible wrapper)
    during their __init__ after calling super().__init__().

    Optional override: _prepare_batch(batch) -> batch
        Called at the start of training_step and validation_step.
        Use this to inject or transform batch fields before the forward pass
        (e.g. setting sc_ca_t for conditional training).
        Default implementation is a no-op.
    """

    def __init__(self, lr=1e-4, rot_loss_weight=1.0,
                 trans_loss_weight=1.0, psi_loss_weight=1.0):
        super().__init__()
        self.lr = lr
        self.rot_loss_weight = rot_loss_weight
        self.trans_loss_weight = trans_loss_weight
        self.psi_loss_weight = psi_loss_weight

    def _prepare_batch(self, batch):
        """Pre-process batch before the forward pass. Override in subclasses."""
        return batch

    def _compute_loss(self, batch):
        """Scale-normalised MSE loss over rotation score, translation score, and psi torsion."""
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

        # Psi torsion: index 2 of torsion_angles_sin_cos; pred['psi'] is [B, N, 2]
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :].float()  # [B, N, 2]
        psi_mse = F.mse_loss(pred['psi'], gt_psi, reduction='none').sum(dim=-1)  # [B, N]
        psi_loss = (psi_mse * mask).sum() / n_visible

        total = (self.rot_loss_weight * rot_loss
                 + self.trans_loss_weight * trans_loss
                 + self.psi_loss_weight * psi_loss)
        return total, rot_loss, trans_loss, psi_loss

    def training_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch)
        loss, rot_loss, trans_loss, psi_loss = self._compute_loss(batch)
        self.log('train_loss',       loss,       on_step=True,  on_epoch=True,  prog_bar=True)
        self.log('train_rot_loss',   rot_loss,   on_step=False, on_epoch=True)
        self.log('train_trans_loss', trans_loss, on_step=False, on_epoch=True)
        self.log('train_psi_loss',   psi_loss,   on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch)
        loss, rot_loss, trans_loss, psi_loss = self._compute_loss(batch)
        self.log('val_loss',         loss,       on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_rot_loss',     rot_loss,   on_step=False, on_epoch=True)
        self.log('val_trans_loss',   trans_loss, on_step=False, on_epoch=True)
        self.log('val_psi_loss',     psi_loss,   on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
