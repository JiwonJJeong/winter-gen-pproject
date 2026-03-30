"""StarScoreNetwork — ScoreNetwork extended for STAR-MD multi-frame training.

Replaces the upstream ScoreNetwork for all training runs.
Single-frame behaviour (rigids_t.ndim == 3, star.enabled=False) is identical
to the upstream ScoreNetwork — this is a drop-in replacement.

Key differences vs. upstream ScoreNetwork:
  1. Uses StarIpaScore instead of IpaScore.
  2. Builds an AdaLN conditioning vector cond = [t_emb ∥ delta_t_emb] and
     passes it to StarIpaScore.
  3. For multi-frame batches (rigids_t.ndim == 4 → [B, L, N, 7]):
       - calls Embedder once per frame with that frame's sc_ca_t [:, l, :, :]
       - stacks results to [B, L, N, c_s] / [B, L, N, N, c_z]
       - forwards frame_idx and cond to StarIpaScore
       - skips the expensive backbone (atom37/atom14) computation
"""

import torch
import torch.nn as nn

import gen_model.path_setup  # noqa: F401  — adds extern/ dirs to sys.path
from model.score_network import Embedder, get_timestep_embedding, get_index_embedding
from data import all_atom

from gen_model.models.star_ipa import StarIpaScore


class StarScoreNetwork(nn.Module):
    """Score network with STAR-MD spatio-temporal attention.

    Args:
        model_conf: OmegaConf node (same schema as default_model_conf).
            If model_conf.star.enabled is False, behaves identically to the
            upstream ScoreNetwork (single-frame, no cross-frame attention).
        diffuser: SE3Diffuser instance.
    """

    def __init__(self, model_conf, diffuser):
        super().__init__()
        self._model_conf = model_conf
        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = StarIpaScore(model_conf, diffuser)

    # ------------------------------------------------------------------
    # AdaLN conditioning vector
    # ------------------------------------------------------------------

    def _build_cond(self, input_feats: dict) -> torch.Tensor:
        """Build [B, 2*D] conditioning vector from diffusion time and delta_t.

        cond = cat([t_emb(t), index_emb(log_delta_t_normalised)], dim=-1)

        For unconditional batches (no delta_t in input_feats) the second half
        is zeros so AdaLN reduces to an identity residual (gamma=1, beta=0 at
        init, no delta_t signal).
        """
        D = self._model_conf.embed.index_embed_size  # 32
        t_emb = get_timestep_embedding(input_feats['t'], D)  # [B, D]

        delta_t = input_feats.get('delta_t', None)
        if delta_t is not None:
            # LogUniform[0.01, 10] ns → log range [-4.61, 2.30]
            # Normalise to zero-mean, unit-std: midpoint -1.155, half-width 3.456
            log_dt = torch.log(delta_t.float().clamp(min=1e-3))  # [B]
            log_dt_norm = (log_dt - (-1.155)) / 3.456
            dt_emb = get_index_embedding(log_dt_norm, embed_size=D)  # [B, D]
        else:
            dt_emb = torch.zeros(
                input_feats['t'].shape[0], D,
                device=input_feats['t'].device,
                dtype=torch.float32,
            )

        return torch.cat([t_emb, dt_emb], dim=-1)  # [B, 2*D]

    # ------------------------------------------------------------------
    # Psi masking helper (mirrors upstream ScoreNetwork)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_mask(aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_feats: dict, kv_caches=None) -> dict:
        """Compute reverse-diffusion scores for a batch.

        Handles both:
          - Single-frame: rigids_t [B, N, 7], sc_ca_t [B, N, 3]
          - Multi-frame:  rigids_t [B, L, N, 7], sc_ca_t [B, L, N, 3]

        Returns:
            dict with keys:
              psi          [B, N, 2] or [B, L, N, 2]
              rot_score    [B, N, 3] or [B, L, N, 3]
              trans_score  [B, N, 3] or [B, L, N, 3]
              rigids       [B, N, 7] or [B, L, N, 7]
              atom37 / atom14  only for single-frame
        """
        bb_mask    = input_feats['res_mask'].type(torch.float32)     # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)   # [B, N]
        edge_mask  = bb_mask[..., None] * bb_mask[..., None, :]      # [B, N, N]

        # AdaLN conditioning vector
        cond = self._build_cond(input_feats)   # [B, 2*D]

        # Multi-frame detection
        multi_frame = input_feats['rigids_t'].ndim == 4

        if multi_frame:
            B, L, N, _ = input_feats['rigids_t'].shape
            sc_ca_t  = input_feats['sc_ca_t']                   # [B, L, N, 3]
            frame_idx = input_feats.get('frame_idx', None)
            # DataLoader collates [L] → [B, L]; extract first item since
            # frame indices are the same across the batch.
            if frame_idx is not None and frame_idx.ndim == 2:
                frame_idx = frame_idx[0]                          # [L]

            node_embeds, edge_embeds = [], []
            for l in range(L):
                n_emb, e_emb = self.embedding_layer(
                    seq_idx=input_feats['seq_idx'],
                    t=input_feats['t'],
                    fixed_mask=fixed_mask,
                    self_conditioning_ca=sc_ca_t[:, l],
                )
                node_embeds.append(n_emb * bb_mask[..., None])
                edge_embeds.append(e_emb * edge_mask[..., None])

            init_node_embed = torch.stack(node_embeds, dim=1)   # [B, L, N, c_s]
            init_edge_embed = torch.stack(edge_embeds, dim=1)   # [B, L, N, N, c_z]
        else:
            frame_idx = None
            sc_ca_t   = input_feats['sc_ca_t']                  # [B, N, 3]

            init_node_embed, init_edge_embed = self.embedding_layer(
                seq_idx=input_feats['seq_idx'],
                t=input_feats['t'],
                fixed_mask=fixed_mask,
                self_conditioning_ca=sc_ca_t,
            )
            init_node_embed = init_node_embed * bb_mask[..., None]
            init_edge_embed = init_edge_embed * edge_mask[..., None]

        # Run score model (StarIpaScore)
        model_out = self.score_model(
            init_node_embed, init_edge_embed, input_feats,
            frame_idx=frame_idx, cond=cond, kv_caches=kv_caches,
        )

        # Psi angle prediction with fixed-residue masking
        gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :].float()
        if multi_frame:
            # fixed_mask [B, N] → expand to [B, L, N, 1]
            fm_exp = fixed_mask[:, None, :, None].expand_as(model_out['psi'])
            psi_pred = self._apply_mask(model_out['psi'], gt_psi, 1 - fm_exp)
        else:
            psi_pred = self._apply_mask(
                model_out['psi'], gt_psi, 1 - fixed_mask[..., None])

        pred_out = {
            'psi':         psi_pred,
            'rot_score':   model_out['rot_score'],
            'trans_score': model_out['trans_score'],
            'rigids':      model_out['final_rigids'].to_tensor_7(),
        }

        # Backbone atom positions — only for single-frame (not needed for training loss)
        if not multi_frame:
            rigids_pred = model_out['final_rigids']
            bb_repr = all_atom.compute_backbone(rigids_pred, psi_pred)
            pred_out['atom37'] = bb_repr[0].to(rigids_pred.device)
            pred_out['atom14'] = bb_repr[-1].to(rigids_pred.device)

        return pred_out
