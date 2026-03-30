"""Optuna hyperparameter optimisation for SE(3) diffusion training.

Searches over:
  - lr                   : learning rate (log-uniform)
  - seq_tfmr_num_layers  : transformer layers per IPA block (int 1–4)

Boundary values (min_sigma, max_sigma, min_b, max_b) are kept fixed at their
physically motivated defaults.  Only the *shape* of the schedule is searched.

The study is stored in an SQLite file so it can be resumed and run with
multiple parallel workers (``--n_jobs`` in study.optimize).

Pruner options (--pruner):
  none  (default) : run all trials to completion — useful for initial exploration
  asha            : Asynchronous Successive Halving (ASHA) via Optuna's
                    SuccessiveHalvingPruner.  Terminates unpromising trials early
                    based on intermediate val_loss reported each epoch.
                    Tune with --asha_min_resource and --asha_reduction_factor.

Usage:
    # No pruning (baseline exploration)
    python gen_model/hpo.py --mode unconditional --data_dir data --n_trials 20

    # ASHA: keep top 1/3 at each rung, first rung at epoch 2
    python gen_model/hpo.py --mode unconditional --data_dir data --n_trials 50 \\
        --pruner asha --asha_min_resource 2 --asha_reduction_factor 3

    # Resume a previous study (load_if_exists=True keeps all past trials):
    python gen_model/hpo.py --mode unconditional --data_dir data --n_trials 10
"""
import os
import argparse
import torch
import optuna
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

def _suggest_hparams(trial: optuna.Trial) -> dict:
    """Sample all hyperparameters for one Optuna trial.

    lora_r is fixed to 0 (full fine-tuning): HPO data showed r=0 consistently
    outperforms all LoRA ranks (r=4/8/16) on short 5-epoch trials.

    Loss weights are fixed at physically motivated defaults (1/1/0.5): rotation
    and translation scores are already normalised by their score_scaling factors
    so equal weighting is principled; psi is an auxiliary signal so 0.5.
    """
    return {
        # lr is intentionally NOT searched here — the LR finder in the final
        # training cell finds the optimal rate for the full model.  Searching
        # lr with only 5-epoch trials introduces a short-horizon bias that
        # systematically favours conservative rates that are too slow for the
        # full 100-epoch run.  3e-4 is a reasonable default for HPO trials.
        'lr':                   3e-4,
        'seq_tfmr_num_layers':  trial.suggest_int('seq_tfmr_num_layers', 1, 2),
        'lora_r':               0,
        'lora_alpha':           0.0,
        'rot_loss_weight':      1.0,
        'trans_loss_weight':    1.0,
        'psi_loss_weight':      0.5,
    }


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _build_se3_conf(hp: dict, cache_dir: str = '/tmp/igso3_cache'):
    """Build SE3Diffuser config.  Schedule curvature is fixed at 1.0 (default).
    Boundary values are physically motivated and kept fixed across all trials.

    Args:
        cache_dir: IGSO3 lookup-table cache directory.  Pass a Drive path to
            persist the cache across Colab sessions (~10 min first-time cost).
    """
    from omegaconf import OmegaConf
    return OmegaConf.create({
        'diffuse_rot': True, 'diffuse_trans': True,
        'so3': {
            'schedule': 'logarithmic',
            'min_sigma': 0.1, 'max_sigma': 1.5,
            'num_sigma': 1000, 'use_cached_score': False,
            'cache_dir': cache_dir, 'num_omega': 1000,
            'schedule_gamma': 1.0,
        },
        'r3': {
            'min_b': 0.1, 'max_b': 20.0,
            'schedule_gamma': 1.0,
        },
    })


# ---------------------------------------------------------------------------
# Dataset / module builders
# ---------------------------------------------------------------------------

def _build_datasets(args, data_args, diffuser):
    """Instantiate train and val datasets for the chosen mode."""
    if args.mode == 'conditional':
        from gen_model.data.dataset import ConditionalMDGenDataset
        train = ConditionalMDGenDataset(
            args=data_args, diffuser=diffuser, mode='train',
            max_k=1, current_max_k=1,   # fixed k=±1 during short HPO trials
        )
        val = ConditionalMDGenDataset(
            args=data_args, diffuser=diffuser, mode='val',
            max_k=1, current_max_k=1,
        )
    else:
        from gen_model.data.dataset import MDGenDataset
        train = MDGenDataset(args=data_args, diffuser=diffuser, mode='train')
        val   = MDGenDataset(args=data_args, diffuser=diffuser, mode='val')
    val.coord_scale = float(train.coord_scale)
    return train, val


def _build_module(mode: str, model_conf, se3_conf, hp: dict):
    """Instantiate the Lightning module for the chosen training mode."""
    kw = dict(
        model_conf=model_conf, se3_conf=se3_conf,
        lr=hp['lr'],
        rot_loss_weight=hp['rot_loss_weight'],
        trans_loss_weight=hp['trans_loss_weight'],
        psi_loss_weight=hp['psi_loss_weight'],
    )
    if mode == 'conditional':
        from gen_model.train_conditional import ConditionalSE3Module
        return ConditionalSE3Module(**kw)
    from gen_model.train_unconditional import SE3Module
    return SE3Module(**kw)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(args):
    """Return an Optuna objective function closed over parsed CLI args."""

    def objective(trial: optuna.Trial) -> float:
        import pytorch_lightning as L
        from pytorch_lightning.callbacks import ModelCheckpoint
        from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
        from gen_model.train_base import default_model_conf, default_data_args
        from gen_model.diffusion.se3_diffuser import SE3Diffuser

        hp         = _suggest_hparams(trial)
        se3_conf   = _build_se3_conf(hp, cache_dir=args.igso3_cache_dir)
        model_conf = default_model_conf(
            use_temporal_embedding=(args.mode == 'conditional'),
            lora_r=hp['lora_r'],
            lora_alpha=hp['lora_alpha'],
            local_attn_sigma=args.local_attn_sigma,
            seq_tfmr_sigma=args.seq_tfmr_sigma,
            seq_tfmr_num_layers=hp['seq_tfmr_num_layers'],
        )
        data_args = default_data_args(args)
        diffuser  = SE3Diffuser(se3_conf)

        train_ds, val_ds = _build_datasets(args, data_args, diffuser)
        pl_module        = _build_module(args.mode, model_conf, se3_conf, hp)

        # PyTorchLightningPruningCallback reports val_loss to the trial after
        # each validation epoch and raises TrialPruned when the pruner decides
        # the trial is unpromising.  With NopPruner (default) it only reports.
        pruning_cb = PyTorchLightningPruningCallback(trial, monitor='val_loss')
        ckpt_cb = ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, f'trial_{trial.number}'),
            filename='best', monitor='val_loss', save_top_k=1, mode='min',
        )

        _nw = 2   # parallel data-loading workers; reduces GPU idle time
        _pm = torch.cuda.is_available()
        trainer = L.Trainer(
            max_epochs=args.epochs_per_trial,
            accelerator='auto', devices='auto',
            callbacks=[pruning_cb, ckpt_cb],
            precision='16-mixed' if torch.cuda.is_available() else 32,
            enable_progress_bar=False,
            logger=False,
            limit_val_batches=0.5,   # HPO needs relative ordering, not exact val_loss
        )

        try:
            trainer.fit(
                pl_module,
                train_dataloaders=DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=_nw, pin_memory=_pm, persistent_workers=_nw > 0),
                val_dataloaders=DataLoader(
                    val_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=_nw, pin_memory=_pm, persistent_workers=_nw > 0),
            )
        except optuna.exceptions.TrialPruned:
            raise  # propagate so Optuna records the pruned state

        return trainer.callback_metrics['val_loss'].item()

    return objective


# ---------------------------------------------------------------------------
# Pruner factory
# ---------------------------------------------------------------------------

def _build_pruner(args) -> optuna.pruners.BasePruner:
    """Construct the Optuna pruner from parsed CLI args.

    'none'  → NopPruner  (no early stopping; all trials run to completion)
    'asha'  → SuccessiveHalvingPruner implementing ASHA.

    ASHA parameters:
        min_resource       : earliest epoch at which a trial can be pruned.
                             Set to ≥ epochs_per_trial/4 so trials get a fair start.
        reduction_factor   : at each rung, keep the top 1/factor trials.
                             factor=3 means ~33% survive each bracket level.
    """
    if args.pruner == 'asha':
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=args.asha_min_resource,
            reduction_factor=args.asha_reduction_factor,
        )
    return optuna.pruners.NopPruner()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run an Optuna study for SE(3) diffusion HPO."""
    parser = argparse.ArgumentParser(description='Optuna HPO for SE(3) Diffusion')
    parser.add_argument('--mode',             type=str,   default='unconditional',
                        choices=['unconditional', 'conditional'],
                        help='Training mode: unconditional or conditional')
    parser.add_argument('--data_dir',         type=str,   default='data')
    parser.add_argument('--atlas_csv',        type=str,   default='gen_model/splits/atlas.csv')
    parser.add_argument('--train_split',      type=str,   default='gen_model/splits/atlas.csv')
    parser.add_argument('--suffix',           type=str,   default='_latent')
    parser.add_argument('--batch_size',       type=int,   default=8)
    parser.add_argument('--n_trials',         type=int,   default=20,
                        help='Number of Optuna trials to run')
    parser.add_argument('--epochs_per_trial', type=int,   default=5,
                        help='Training epochs per trial (keep short for HPO)')
    parser.add_argument('--save_dir',         type=str,   default='checkpoints/hpo',
                        help='Directory for SQLite DB and per-trial checkpoints')
    # Pruner selection
    parser.add_argument('--pruner',               type=str,   default='none',
                        choices=['none', 'asha'],
                        help='Trial pruning strategy (none = run all trials to completion)')
    parser.add_argument('--asha_min_resource',    type=int,   default=1,
                        help='ASHA: minimum epochs before a trial can be pruned')
    parser.add_argument('--asha_reduction_factor',type=int,   default=3,
                        help='ASHA: keep top 1/reduction_factor trials at each rung')
    # Attention cutoffs — keep consistent with full training so HPO results transfer.
    parser.add_argument('--local_attn_sigma',     type=float, default=15.0,
                        help='IPA spatial attention cutoff in Å (~2%% at this distance). 0=global.')
    parser.add_argument('--seq_tfmr_sigma',       type=float, default=20.0,
                        help='Seq-transformer attention cutoff in residues. 0=global.')
    parser.add_argument('--igso3_cache_dir',      type=str,   default='/tmp/igso3_cache',
                        help='Directory for the IGSO3 SO3 lookup-table cache.  Pass a Drive path '
                             'to persist across Colab sessions (~10 min first-time cost).')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    pruner = _build_pruner(args)
    study = optuna.create_study(
        direction='minimize',
        storage=f'sqlite:///{args.save_dir}/optuna.db',
        study_name=f'se3_{args.mode}',
        load_if_exists=True,      # resume seamlessly if the study already exists
        pruner=pruner,
    )
    study.optimize(
        make_objective(args),
        n_trials=args.n_trials,
        catch=(Exception,),   # log failures as failed trials rather than crashing
    )

    pruner_name = 'ASHA' if args.pruner == 'asha' else 'none'
    print(f'\n=== HPO complete: {len(study.trials)} total trials | pruner={pruner_name} ===')
    best = study.best_trial
    print(f'Best val_loss : {best.value:.6f}  (trial #{best.number})')
    print('Best params:')
    for k, v in best.params.items():
        print(f'  {k:25s}: {v}')


if __name__ == '__main__':
    main()
