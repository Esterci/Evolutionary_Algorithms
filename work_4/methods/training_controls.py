"""Shared Adam scheduling and framework early-stopping configuration.

The adaptive stopping metric is the unweighted training MSE sum, not the
regularized objective and not held-out validation error. Final iterates are
retained, matching the framework's checkpoint semantics.
"""

import argparse
import inspect
import math
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler


class LogarithmicLR(LRScheduler):
    """Decay from base LR to final_ratio * base LR over the planned updates.

    For update index j (zero based), factor = 1 - (1-r) *
    log(1 + min(j, E-1)) / log(E). A one-update schedule uses the base LR.
    Trainer advances the scheduler after each optimizer step.
    """

    def __init__(self, optimizer, epochs, final_ratio=0.1, last_epoch=-1):
        if epochs < 1 or not math.isfinite(final_ratio) or not 0 < final_ratio <= 1:
            raise ValueError('Require epochs >= 1 and 0 < final_ratio <= 1')
        self.epochs = epochs
        self.final_ratio = final_ratio
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        progress = (math.log1p(min(max(self.last_epoch, 0), self.epochs - 1)) /
                    math.log(self.epochs)) if self.epochs > 1 else 0.0
        factor = 1.0 - (1.0 - self.final_ratio) * progress
        return [base_lr * factor for base_lr in self.base_lrs]


def add_control_arguments(parser):
    parser.add_argument('--scheduler', choices=['logarithmic', 'none'], default='logarithmic')
    parser.add_argument('--lr-final-ratio', type=float, default=0.1,
                        help='Final/initial Adam LR ratio over the planned Adam updates')
    parser.add_argument('--early-stopping', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--early-stopping-patience', type=int, default=300)
    parser.add_argument('--early-stopping-tolerance', type=float, default=1e-5,
                        help='Minimum absolute improvement of the framework training monitor')


def control_config(args, parser=None):
    valid = (math.isfinite(args.lr) and args.lr > 0 and
             math.isfinite(args.lr_final_ratio) and 0 < args.lr_final_ratio <= 1 and
             args.early_stopping_patience >= 1 and
             math.isfinite(args.early_stopping_tolerance) and args.early_stopping_tolerance >= 0)
    if not valid:
        message = 'Require finite lr > 0, 0 < lr-final-ratio <= 1, patience >= 1 and finite tolerance >= 0'
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return dict(scheduler=args.scheduler, final_ratio=args.lr_final_ratio,
                early_stopping=args.early_stopping, patience=args.early_stopping_patience,
                tolerance=args.early_stopping_tolerance)


def require_training_control_api():
    """Reject old frameworks before starting a potentially expensive DE stage."""
    from methods.pinn_model import Trainer

    if 'scheduler' not in inspect.signature(Trainer).parameters:
        raise RuntimeError('Updated Pinn-Torch Trainer required for scheduling/early stopping. '
                           'Set FISIOCOMPINN_PATH to the updated framework checkout.')


def trainer_control_kwargs(optimizer, epochs, config):
    if config is None:
        # Low-level helpers retain fixed-budget behavior unless configured.
        return dict(patience=epochs, tolerance=0.0)
    
    require_training_control_api()
    
    scheduler = (LogarithmicLR(optimizer, epochs, config['final_ratio'])
                 if config['scheduler'] == 'logarithmic' else None)
    return dict(scheduler=scheduler,
                patience=config['patience'] if config['early_stopping'] else None,
                tolerance=config['tolerance'])


def save_training_controls(trainer, config, directory, de_evaluations=0):
    """Save actual Adam LR/monitor trajectories and effective training budget."""
    directory = Path(directory)
    rates = np.asarray(trainer.learning_rate_history, dtype=float)
    monitor = np.asarray(trainer.monitor_history, dtype=float)
    np.savez_compressed(directory / 'adam_training_controls.npz',
                        learning_rates=rates, monitor=monitor,
                        epoch=np.arange(1, len(rates) + 1))
    if trainer.scheduler is not None:
        torch.save(trainer.scheduler.state_dict(), directory / 'adam_scheduler.pt')
    return dict(
        training_controls=config,
        adam_epochs_run=trainer.n_epochs_run,
        training_loss_evaluations=de_evaluations + trainer.n_epochs_run,
        early_stopping=dict(enabled=config['early_stopping'],
                            stopped_early=trainer.stopped_early,
                            patience=trainer.patience, tolerance=trainer.tolerance,
                            monitor='sum of raw training MSEs' if trainer.adaptive
                            else 'fixed weighted training MSE sum',
                            timing='pre-update; checked after the optimizer step',
                            best_loss=trainer.best_loss,
                            restored_best_weights=False),
        learning_rate_schedule=dict(name=config['scheduler'], planned_adam_updates=trainer.n_it,
                                    final_ratio=config['final_ratio'],
                                    first_used=rates[0].tolist(), last_used=rates[-1].tolist(),
                                    next_step=[group['lr'] for group in trainer.optimizer.param_groups]),
    )
