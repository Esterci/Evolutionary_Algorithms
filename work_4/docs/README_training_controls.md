# Adam learning-rate decay and early stopping

Both neural-network runners enable these controls by default. They use the
updated FisiocomPINN Trainer directly; DE keeps its configured generation count.
No equations, collocation grids, initial/boundary conditions, or adaptive loss
formula are changed.

## Framework requirement

Validated against Pinn-Torch commit `abb148e8ad0857a5c49abcfd65163d040818bd11`.
The Trainer must accept `optimizer` and `scheduler` and implement monitoring in
both training loops. The older sibling checkout is not sufficient. To select
an updated checkout explicitly, run from `work_4`:

```bash
export FISIOCOMPINN_PATH=/path/to/updated/Pinn-Torch
python -m runners.train_pinn --epochs 5000
python -m runners.train_pinn_de_adam --epochs 4932 --de-generations 100 --population-size 32
```

An obsolete API raises an actionable error instead of silently ignoring controls.
The framework checkout itself is not modified by this implementation.

## Logarithmic schedule

For zero-based Adam update index `j`, planned Adam updates `E > 1`, initial
learning rate `lr0`, and final ratio `r`:

```text
lr(j) = lr0 * [1 - (1-r) * log(1 + min(j, E-1)) / log(E)]
```

Defaults: `--scheduler logarithmic`, `--lr 0.001`, `--lr-final-ratio 0.1`.
The first update uses 0.001, and the last planned update uses 0.0001. The
framework steps the scheduler after each optimizer update. One-update runs use
the initial LR. Early termination need not reach the target final rate.
For DE/Adam, `E = epochs - de_generations`; the scheduler starts after DE.
The network and adaptive log variances share the same optimizer group and LR.
The decay is logarithmic (not exponential interpolation on a log-scale axis).

## Early stopping

Defaults: `--early-stopping`, `--early-stopping-patience 300`, and
`--early-stopping-tolerance 1e-5`. A decrease strictly greater than this absolute
tolerance resets patience. The first observation establishes the baseline.

- Adaptive training monitors `MSE_initial + MSE_boundary + MSE_PDE`.
- Fixed-weight training monitors `10*MSE_initial + MSE_boundary + MSE_PDE`.
- Metrics are measured before the update, with the stop checked after it.
- The framework returns the final iterate and does not restore best weights.
- Monitoring reuses training losses; it introduces no FVM targets or additional
  validation evaluations. This is a training convergence criterion, not a
  generalization guarantee. Final independent physics/FVM evaluation remains.

The [latest recorded study](../studies/pinn_pair_de32_g100_1788968487925081526/README.md)
reports late growth of initial-condition loss while boundary/PDE losses decline.
It also demonstrates that the regularized adaptive objective can decrease
without better physical agreement. This motivates monitoring raw losses rather
than that objective. The summed monitor can still hide deterioration in one
term; it does not solve the adaptive objective's underlying degeneracy.
The chosen decay, patience, and tolerance are initial experimental settings,
not tuned optima. No full experiment demonstrating improvement has been run.

To reproduce the former constant-LR, fixed-budget behavior:

```bash
python -m runners.train_pinn --scheduler none --no-early-stopping
python -m runners.train_pinn_de_adam --scheduler none --no-early-stopping
```

Both paired shell launchers now inherit the new defaults. Historical artifacts
retain their original settings; they are not overwritten or relabeled.

## Recorded artifacts and budgets

`adam_training_controls.npz` stores the LR actually used for each optimizer
update, the pre-update monitor, and one-based epoch indices. `adam_scheduler.pt`
stores scheduler state alongside the existing model, optimizer, and adaptive
states. Saved state enables inspection but does not add a resume-training CLI.

Metadata retains the planned `epochs` / `adam_epochs` and adds
`adam_epochs_run`, `training_loss_evaluations`, the stopping decision, the
monitor definition, and first/last used LR. DE/Adam's actual evaluation count is
`DE fitness evaluations + completed Adam updates`. Diagnostic evaluations are
outside this count. Analysis tools prefer actual counts, with legacy fallback.
Paired runs now share maximum budgets; early stopping can produce unequal actual
budgets. Report these differences in comparisons with historical experiments.
