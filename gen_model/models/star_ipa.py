"""StarIpaScore: IpaScore extended with STAR-MD spatio-temporal attention.

Extends the upstream SE3-diffusion IpaScore
(extern/se3_diffusion/model/ipa_pytorch.py) without modifying it.

Changes vs. upstream IpaScore:
  1. __init__: injects one SpatioTemporalAttention block per trunk block when
     model_conf.star.enabled is True.  Removes the upstream SeqTransformer
     (skip_embed, seq_tfmr, post_tfmr, node_transition) that is redundant
     with the joint spatio-temporal attention.
  2. forward: adds frame_idx / cond kwargs and multi-frame flatten/unflatten
     logic so the model operates jointly over L MD frames.
  3. KV-cache support for efficient autoregressive inference.

Single-frame behaviour (L=1 or frame_idx=None) is identical to the upstream
IpaScore — StarIpaScore is a drop-in replacement.
"""

import torch
import torch.nn as nn
import gen_model.path_setup  # noqa: F401  — adds extern/se3_diffusion to sys.path

from model.ipa_pytorch import IpaScore
from openfold.utils.rigid_utils import Rigid
from gen_model.models.star_attention import SpatioTemporalAttention


class StarIpaScore(IpaScore):
    """IpaScore with optional STAR-MD spatio-temporal cross-frame attention.

    When star.enabled is True, the upstream per-frame SeqTransformer is
    replaced by joint spatio-temporal attention (following the STAR-MD paper:
    IPA → ST Attention → Pair Update + Coord Prediction).

    Args:
        model_conf: OmegaConf node.  If model_conf.star.enabled is True, one
            SpatioTemporalAttention block is injected into self.trunk after
            each IPA block, and the upstream SeqTransformer modules are removed.
        diffuser:   SE3Diffuser instance (passed through to IpaScore).
    """

    def __init__(self, model_conf, diffuser):
        super().__init__(model_conf, diffuser)

        star_conf = getattr(model_conf, 'star', None)
        self._star_enabled = star_conf is not None and star_conf.enabled

        if self._star_enabled:
            ipa_conf = model_conf.ipa
            cond_dim = 2 * model_conf.embed.index_embed_size  # 2*32 = 64

            spatial_sigma = float(getattr(star_conf, 'spatial_sigma', 0.0))
            for b in range(ipa_conf.num_blocks):
                # Inject ST attention
                self.trunk[f'st_attn_{b}'] = SpatioTemporalAttention(
                    c_s=ipa_conf.c_s,
                    num_heads=star_conf.st_num_heads,
                    cond_dim=cond_dim,
                    causal=star_conf.causal,
                    spatial_sigma=spatial_sigma,
                )
                # Remove upstream SeqTransformer modules (redundant with ST attn)
                for key in [f'skip_embed_{b}', f'seq_tfmr_{b}',
                            f'post_tfmr_{b}', f'node_transition_{b}']:
                    if key in self.trunk:
                        del self.trunk[key]

    # ------------------------------------------------------------------
    # Forward — adds multi-frame support on top of the upstream logic.
    # ------------------------------------------------------------------

    def forward(self, init_node_embed, edge_embed, input_feats,
                frame_idx=None, cond=None, kv_caches=None):
        """
        Args:
            init_node_embed: [B, N, c_s]  or  [B, L, N, c_s]
            edge_embed:      [B, N, N, c_z]  or  [B, L, N, N, c_z]
            input_feats:     dict with keys rigids_t, res_mask, fixed_mask, t, …
                             rigids_t is [B, N, 7] (single) or [B, L, N, 7] (multi).
            frame_idx:       [L] integer frame positions for RoPE; None for single-frame.
            cond:            [B, 2*D] AdaLN conditioning (t_emb ∥ delta_t_emb);
                             None → zeros (no temporal conditioning).
            kv_caches:       Optional list of dicts (one per trunk block) for
                             KV-cache during autoregressive inference.  Each dict
                             is passed to SpatioTemporalAttention and updated
                             in-place.  None → no caching (training mode).

        Returns:
            dict with psi, rot_score, trans_score, final_rigids.
            Scores are [B, N, 3] for single-frame, [B, L, N, 3] for multi-frame.
        """
        # ── Multi-frame detection ──────────────────────────────────────────
        input_rigids_raw = input_feats['rigids_t']
        multi_frame = input_rigids_raw.ndim == 4
        if multi_frame:
            B, L, N = input_rigids_raw.shape[:3]

            def flat(x):
                return x.reshape(B * L, *x.shape[2:])

            def unflat(x):
                return x.reshape(B, L, *x.shape[1:])

            init_node_embed = flat(init_node_embed)   # [B*L, N, c_s]
            edge_embed      = flat(edge_embed)        # [B*L, N, N, c_z]
            # res_mask / fixed_mask are [B, N] — same crop for all frames
            _res_mask   = input_feats['res_mask'].float()[:, None, :].expand(B, L, N)
            _fixed_mask = input_feats['fixed_mask'].float()[:, None, :].expand(B, L, N)
            node_mask       = flat(_res_mask)         # [B*L, N]
            fixed_mask_flat = flat(_fixed_mask)       # [B*L, N]
            rigids_t_flat   = flat(input_rigids_raw)  # [B*L, N, 7]
            # t is [B]; broadcast to all L frames → [B*L]
            t_flat = input_feats['t'].unsqueeze(1).expand(B, L).reshape(B * L)
        else:
            node_mask       = input_feats['res_mask'].type(torch.float32)
            fixed_mask_flat = input_feats['fixed_mask'].type(torch.float32)
            rigids_t_flat   = input_feats['rigids_t'].type(torch.float32)
            t_flat          = input_feats['t']

        diffuse_mask = (1 - fixed_mask_flat) * node_mask
        edge_mask    = node_mask[..., None] * node_mask[..., None, :]
        init_frames  = rigids_t_flat.type(torch.float32)

        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        init_rigids = Rigid.from_tensor_7(init_frames)

        # Sequence transformer Gaussian bias (only when STAR-MD is disabled).
        _seq_tfmr_mask = None
        if not self._star_enabled:
            _seq_cutoff = float(getattr(self._ipa_conf, 'seq_tfmr_sigma', 0.0))
            if _seq_cutoff > 0.0:
                num_res = node_mask.shape[-1]
                idx = torch.arange(num_res, device=node_mask.device, dtype=torch.float32)
                seq_dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
                _seq_tfmr_mask = -(2.0 * seq_dist / _seq_cutoff) ** 2

        # ── Main trunk ────────────────────────────────────────────────────
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed      = init_node_embed * node_mask[..., None]

        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed, edge_embed, curr_rigids, node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)

            # ── Per-frame SeqTransformer (upstream, only when STAR disabled) ─
            if not self._star_enabled:
                seq_tfmr_in = torch.cat([
                    node_embed, self.trunk[f'skip_embed_{b}'](init_node_embed)
                ], dim=-1)
                seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                    seq_tfmr_in, mask=_seq_tfmr_mask,
                    src_key_padding_mask=1 - node_mask)
                node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
                node_embed = self.trunk[f'node_transition_{b}'](node_embed)
                node_embed = node_embed * node_mask[..., None]

            # ── Cross-frame attention (STAR-MD) ───────────────────────────
            if multi_frame and f'st_attn_{b}' in self.trunk:
                cond_dim = self.trunk[f'st_attn_{b}'].adaln.gamma_mlp[0].in_features
                st_cond = cond if cond is not None else torch.zeros(
                    B, cond_dim, device=node_embed.device, dtype=node_embed.dtype)
                st_mask = unflat(node_mask)        # [B, L, N]
                nef     = unflat(node_embed)       # [B, L, N, c_s]

                # CA positions for spatial Gaussian bias (if spatial_sigma > 0)
                ca_pos = unflat(curr_rigids.get_trans())  # [B, L, N, 3]

                cache_b = kv_caches[b] if kv_caches is not None else None
                delta, new_kv_b = self.trunk[f'st_attn_{b}'](
                    nef, frame_idx, input_feats['seq_idx'], st_mask, st_cond,
                    kv_cache=cache_b, ca_pos=ca_pos)
                node_embed = flat(nef + delta)     # residual → [B*L, N, c_s]
                node_embed = node_embed * node_mask[..., None]

            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update * diffuse_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # ── Score computation ─────────────────────────────────────────────
        rot_score = self.diffuser.calc_rot_score(
            init_rigids.get_rots(), curr_rigids.get_rots(), t_flat)
        rot_score = rot_score * node_mask[..., None]

        trans_score = self.diffuser.calc_trans_score(
            init_rigids.get_trans(), curr_rigids.get_trans(),
            t_flat[:, None, None], use_torch=True)
        trans_score = trans_score * node_mask[..., None]

        _, psi_pred = self.torsion_pred(node_embed)

        if multi_frame:
            rot_score   = unflat(rot_score)    # [B, L, N, 3]
            trans_score = unflat(trans_score)  # [B, L, N, 3]
            psi_pred    = unflat(psi_pred)     # [B, L, N, 2]
            curr_rigids = Rigid.from_tensor_7(unflat(curr_rigids.to_tensor_7()))

        return {
            'psi':          psi_pred,
            'rot_score':    rot_score,
            'trans_score':  trans_score,
            'final_rigids': curr_rigids,
        }
