"""Shared base Lightning module and default configurations for SE(3) diffusion.

Both unconditional and conditional DDPM inherit from SE3BaseModule.
Subclasses only need to set self.model and optionally override _prepare_batch.

Config helpers (default_se3_conf, default_model_conf, default_data_args) provide
the canonical hyperparameters shared across all train scripts.
"""
import torch
import torch.nn.functional as F
import lightning as L


# ---------------------------------------------------------------------------
# Shared default configurations
# ---------------------------------------------------------------------------

def default_se3_conf(cache_dir: str = '/tmp/igso3_cache'):
    """Canonical SE(3) diffusion config: logarithmic SO3 schedule + R3 VP-SDE.

    Args:
        cache_dir: Directory for the IGSO3 lookup-table cache.  Defaults to
            /tmp/igso3_cache (ephemeral).  Pass a Google Drive path to persist
            the cache across Colab sessions and avoid recomputing (~10 min).
    """
    from omegaconf import OmegaConf
    return OmegaConf.create({
        'diffuse_rot': True, 'diffuse_trans': True,
        'so3': {'schedule': 'logarithmic', 'min_sigma': 0.1, 'max_sigma': 1.5,
                'num_sigma': 1000, 'use_cached_score': False,
                'cache_dir': cache_dir, 'num_omega': 1000,
                'schedule_gamma': 1.0},
        'r3':  {'min_b': 0.1, 'max_b': 20.0, 'schedule_gamma': 1.0},
    })


def default_model_conf(use_temporal_embedding: bool = False,
                       lora_r: int = 0, lora_alpha: float = 0.0,
                       local_attn_sigma: float = 0.0,
                       seq_tfmr_sigma: float = 0.0,
                       seq_tfmr_num_layers: int = 2,
                       star_enabled: bool = False,
                       st_num_heads: int = 4):
    """Canonical ScoreNetwork config: 256-dim IPA transformer + 128-dim edges.

    Args:
        use_temporal_embedding: Set True for conditional models to enable ϕ(k)
            inside the Embedder (adds 32-dim temporal gap embedding to ϕ(t)).
        lora_r: LoRA rank.  0 = disabled (full fine-tuning).
        lora_alpha: LoRA scaling factor.  Defaults to 2*lora_r when 0.
        local_attn_sigma: IPA spatial attention cutoff in Ångströms.  Attention
            falls to ~2% at this distance (100% near 0, ~37% at cutoff/2).
            0.0 = disabled (global attention).  Typical: 12–20 Å.
        seq_tfmr_sigma: Sequence transformer attention cutoff in residues.
            Attention falls to ~2% at this sequence separation.
            0.0 = disabled (global attention).  Typical: 16–32.
        seq_tfmr_num_layers: Number of transformer layers inside each IPA block.
            Searched during HPO (range 1–4).  Default 2 matches the paper.
    """
    from omegaconf import OmegaConf
    _r = max(lora_r, 1)  # avoid div-by-zero in alpha default; enabled flag gates usage
    return OmegaConf.create({
        'node_embed_size': 256, 'edge_embed_size': 128,
        'embed': {'index_embed_size': 32, 'embed_self_conditioning': True,
                  'num_bins': 22, 'min_bin': 1e-5, 'max_bin': 5.0,
                  'use_temporal_embedding': use_temporal_embedding},
        'ipa': {'c_s': 256, 'c_z': 128, 'c_hidden': 16, 'no_heads': 12,
                'no_qk_points': 4, 'no_v_points': 8, 'c_skip': 64,
                'num_blocks': 4,
                'seq_tfmr_num_heads': 4, 'seq_tfmr_num_layers': seq_tfmr_num_layers,
                'local_attn_sigma': local_attn_sigma,
                'seq_tfmr_sigma': seq_tfmr_sigma},
        'star': {
            'enabled': star_enabled,
            'st_num_heads': st_num_heads,
            'causal': True,
        },
        'lora': {
            'enabled': lora_r > 0,
            'r': _r,
            'alpha': lora_alpha if lora_alpha > 0.0 else float(2 * _r),
            # IPA attention Q/K/V/O + MLP layers in StructureModuleTransition,
            # TorsionAngles, and ScoreLayer.  Geometric projections (linear_q_points,
            # linear_kv_points, linear_b, down_z) and tiny output heads are excluded.
            # SpatioTemporalAttention projections added for STAR-MD.
            'target_modules': [
                'linear_q', 'linear_kv', 'linear_out',
                'linear_1', 'linear_2',
                'q_proj', 'k_proj', 'v_proj', 'out_proj',
            ],
        },
    })


def default_data_args(args):
    """Build OmegaConf data config from parsed CLI args."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        'data_dir': args.data_dir, 'atlas_csv': args.atlas_csv,
        'train_split': args.train_split, 'suffix': args.suffix,
        'frame_interval': None, 'crop_ratio': 0.95, 'min_t': 0.01,
    })


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
        bs = batch['res_mask'].shape[0]
        self.log('train_loss',       loss,       on_step=True,  on_epoch=True,  prog_bar=True, batch_size=bs)
        self.log('train_rot_loss',   rot_loss,   on_step=False, on_epoch=True,                 batch_size=bs)
        self.log('train_trans_loss', trans_loss, on_step=False, on_epoch=True,                 batch_size=bs)
        self.log('train_psi_loss',   psi_loss,   on_step=False, on_epoch=True,                 batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch)
        loss, rot_loss, trans_loss, psi_loss = self._compute_loss(batch)
        bs = batch['res_mask'].shape[0]
        self.log('val_loss',         loss,       on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log('val_rot_loss',     rot_loss,   on_step=False, on_epoch=True,                batch_size=bs)
        self.log('val_trans_loss',   trans_loss, on_step=False, on_epoch=True,                batch_size=bs)
        self.log('val_psi_loss',     psi_loss,   on_step=False, on_epoch=True,                batch_size=bs)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
