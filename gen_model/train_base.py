"""Shared configuration helpers for SE(3) diffusion training.

Lightning modules have moved to gen_model/se3_diffusion_module.py
following the SinFusion training pattern.
"""


def default_se3_conf(cache_dir: str = '/tmp/igso3_cache'):
    """Canonical SE(3) diffusion config: logarithmic SO3 schedule + R3 VP-SDE."""
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
        local_attn_sigma: IPA spatial attention cutoff in Ångströms.  0 = global.
        seq_tfmr_sigma: Sequence transformer attention cutoff in residues.  0 = global.
        seq_tfmr_num_layers: Transformer layers per IPA block.
        star_enabled: Enable STAR-MD spatio-temporal attention.
        st_num_heads: Number of attention heads in SpatioTemporalAttention.
    """
    from omegaconf import OmegaConf
    _r = max(lora_r, 1)
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
            # IPA attention Q/K/V/O + MLP layers + SpatioTemporalAttention projections.
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
        # Physical time between consecutive stored frames (ns).
        # Set via --ns_per_stored_frame; used by ConditionalMDGenDataset to
        # convert sampled delta_t (ns) to a raw-frame stride k.
        'ns_per_stored_frame': getattr(args, 'ns_per_stored_frame', 0.1),
    })
