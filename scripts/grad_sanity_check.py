"""One-step gradient sanity check on the conditional STAR-MD model — runs on CPU.

Builds a minimum-viable synthetic batch (correctly-shaped tensors with valid
quaternions; not real protein data), runs one forward+backward through the
conditional model, and prints which trainable param has what kind of gradient
(healthy / ZERO / NONE).

Use this whenever stage-2 training behaves strangely (e.g. weights not moving,
"no denoising at inference") to confirm gradient is actually flowing through
every part of the architecture you expect to be learning.

Environment knobs:
    SIMULATE_POST_STAGE_1=1  (default)  Nudge bb_update.linear and
        torsion_pred.linear_final off zero. These are zero-init AND frozen in
        a fresh stage-2 model (not in LoRA target_modules), and a real run
        loads them from a stage-1 checkpoint. Without this nudge, EVERY
        gradient is zero — which is the correct diagnostic for "stage-1 ckpt
        not loaded" but useless for verifying the rest of the architecture.
    SIMULATE_POST_STEP_1=1   (default 0)  Also nudge out_proj.lora_B off
        zero in every ST attention block, simulating the state after the
        first optimizer step. With this, gradients should cascade upstream
        to AdaLN, log_spatial_sigma, and q/k/v projections.

Usage:
    PYTHONPATH=. python scripts/grad_sanity_check.py
    SIMULATE_POST_STEP_1=1 PYTHONPATH=. python scripts/grad_sanity_check.py
"""

import os
import re
import numpy as np
import torch

import gen_model.path_setup  # noqa: F401
from gen_model.train_base import default_se3_conf, default_model_conf
from gen_model.diffusion.se3_diffuser import SE3Diffuser
from gen_model.models.star_score_network import StarScoreNetwork
from gen_model.models.lora import apply_lora
from gen_model.se3_diffusion_module import ConditionalSE3Diffusion


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # Tiny config for CPU speed
    B, L, N = 1, 4, 12

    se3_conf = default_se3_conf(cache_dir='/tmp/igso3_cache')
    model_conf = default_model_conf(
        use_temporal_embedding=True,
        lora_r=8, lora_alpha=16,
        star_enabled=True,
        st_num_heads=8,
        spatial_sigma=8.0,
        num_blocks=2,
    )
    # default_model_conf doesn't set ipa.coordinate_scaling; upstream IPA needs it.
    model_conf.ipa.coordinate_scaling = 0.1

    diffuser = SE3Diffuser(se3_conf)
    score_net = StarScoreNetwork(model_conf, diffuser)
    apply_lora(score_net, model_conf.lora)

    if os.environ.get('SIMULATE_POST_STAGE_1', '1') == '1':
        with torch.no_grad():
            for name, p in score_net.named_parameters():
                if (name.endswith('bb_update_0.linear.weight')
                    or name.endswith('bb_update_1.linear.weight')
                    or name.endswith('torsion_pred.linear_final.weight')):
                    p.normal_(0.0, 0.01)
                    print(f'  nudged {name} to N(0, 0.01)')

    if os.environ.get('SIMULATE_POST_STEP_1', '0') == '1':
        with torch.no_grad():
            for name, p in score_net.named_parameters():
                if name.endswith('out_proj.lora_B') and 'st_attn_' in name:
                    p.normal_(0.0, 0.01)
                    print(f'  nudged {name} to N(0, 0.01)')

    module = ConditionalSE3Diffusion(
        model=score_net, diffuser=diffuser,
        lr=1e-4, ema_decay=0.0,
        weight_log_every_n_steps=1,
        weight_log_print_every_n_steps=1,
    )
    module.eval()

    # Synthetic batch — correct shapes with valid quaternions
    def random_unit_quat(*shape):
        q = torch.randn(*shape, 4)
        return q / q.norm(dim=-1, keepdim=True)

    quats  = random_unit_quat(B, L, N)
    trans  = torch.randn(B, L, N, 3) * 0.5
    rigids_0 = torch.cat([quats, trans], dim=-1)
    torsions = torch.randn(B, L, N, 7, 2)
    torsions = torsions / torsions.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    batch = {
        'rigids_0':                rigids_0,
        'res_mask':                torch.ones(B, N),
        'fixed_mask':              torch.zeros(B, N),
        'aatype':                  torch.zeros(B, N, dtype=torch.long),
        'seq_idx':                 torch.arange(1, N + 1).unsqueeze(0).expand(B, N).contiguous(),
        'chain_idx':               torch.ones(B, N),
        'torsion_angles_sin_cos':  torsions,
        'atom14_pos':              torch.randn(B, L, N, 14, 3) * 0.5,
        'atom37_pos':              torch.randn(B, L, N, 37, 3) * 0.5,
        'atom37_mask':             torch.ones(B, L, N, 37),
        'residx_atom14_to_atom37': torch.zeros(B, L, 14, dtype=torch.long),
        'residue_index':           torch.arange(N).unsqueeze(0).expand(B, N).contiguous(),
        'sc_ca_t':                 torch.randn(B, L, N, 3) * 0.5,
        'frame_idx':               torch.arange(L),
        'delta_t':                 torch.full((B,), 0.1),
        'k':                       torch.ones(B, dtype=torch.long),
    }

    print('=' * 72)
    print('Running forward + backward on synthetic batch...')
    print('=' * 72)

    loss, rot, trans_l, psi = module.forward(batch)
    print(f'loss={loss.item():.4g}  rot={rot.item():.4g}  '
          f'trans={trans_l.item():.4g}  psi={psi.item():.4g}')
    loss.backward()

    print()
    print('=' * 72)
    print('Per-block ST attention gradient table:')
    print('=' * 72)
    print(f'{"param":<70} {"requires_grad":<14} {"grad_status":<14} {"||grad||":>12}')
    print('-' * 110)

    block_re = re.compile(r'.*st_attn_(\d+)\.')
    seen_blocks = set()
    healthy = zero = none_count = 0
    for name, p in module.model.named_parameters():
        m = block_re.match(name)
        if m is None:
            continue
        seen_blocks.add(m.group(1))
        if not p.requires_grad:
            status, gnorm = 'frozen', '—'
        elif p.grad is None:
            status, gnorm = 'NONE', '—'
            none_count += 1
        else:
            gn = p.grad.detach().norm().item()
            if gn == 0.0:
                status, gnorm = 'ZERO', '0'
                zero += 1
            else:
                status, gnorm = 'OK', f'{gn:.4g}'
                healthy += 1
        print(f'{name:<70} {str(p.requires_grad):<14} {status:<14} {gnorm:>12}')

    print('-' * 110)
    print(f'blocks seen: {sorted(seen_blocks)}')
    print(f'healthy (nonzero grad): {healthy}, ZERO grad: {zero}, NONE grad: {none_count}')


if __name__ == '__main__':
    main()
