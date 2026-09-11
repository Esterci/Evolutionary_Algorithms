# Differential Evolution followed by Adam

`train_pinn_de_adam.py` trains the existing `work_4` Burgers PINN in two stages:
DE optimizes all network weights and biases plus three adaptive loss-weight genes
for the initial condition, boundary condition and combined PDE residual. The best
genome initializes Adam, which continues adapting all three weights with the same
objective as the ordinary adaptive PINN.
It uses the local sibling `../Pinn-Torch` checkout when available, including
its `Trainer(..., optimizer=...)` interface. No framework changes are required.

## Run

Activate the environment containing PyTorch and FisiocomPINN, then run from the
repository root:

```bash
conda activate torch-numba-11
python work_4/train_pinn_de_adam.py \
  --epochs 4852 --de-generations 20 --population-size 8 \
  --hidden-sizes 32 32 32 \
  --lower-bound -2 --upper-bound 2 \
  --lr 0.001 --seed 2028 --de-seed 3028
```

This schedule performs **20 complete DE generations followed by 4,832 Adam
epochs**, plus the DE population's initial evaluation. DE therefore uses
`8 * (20 + 1) = 168` fitness evaluations. Each fitness evaluation computes
all full-batch loss terms and includes the required input derivatives, but
does not run Adam or backpropagate to train network parameters. Adam uses one
full-batch backward/update per epoch. Schedule slots are not equal computational
budgets; compare timings, evaluation counts and network sizes as well.

To run both methods with the selected `3-32-32-32-2` topology and 5,000 training
loss evaluations each, use `bash work_4/run_recommended_pair.sh`. It uses the
current physical configuration in `control_dicts/`. Optional environment
variables are `PYTHON_BIN`, `DEVICE`, `NETWORK_SEED` and `OUTPUT_DIRECTORY`.
The DE seed is the network seed plus 1,000. Both training entry points also
default to this topology and budget; the paired script explicitly sets every
selected hyperparameter and uses the first confirmation seed unless overridden.
The topology and learning rate were selected from a small validation screening,
not an exhaustive search. DE population and generation counts were fixed for
that comparison, not independently optimized. See the
[recorded study](studies/pinn_pair_1788963619775820198/README.md) for raw tables,
independent-seed confirmation and numerical-reference limitations.

The requested [32-individual, 100-generation follow-up](studies/pinn_pair_de32_g100_1788968487925081526/README.md)
reruns three paired seeds at 8,064 training loss evaluations per method. Use
`bash work_4/run_de32_pair.sh` to reproduce that larger schedule. Its report
compares it directly with the original 8-individual, 20-generation study and
documents the shared adaptive objective's constant-field counterexample.

`--epochs` must exceed `--de-generations`, which must be at least one. Population
size must be at least four. Use `--device cpu` or `--device cuda:0`, and
`--dtype float64` when required. Both stages share the same dtype and device.
The ordinary `train_pinn.py` entry point uses the same adaptive formula and now
also saves the adaptive state/history. Its framework weight module explicitly
matches the network device and dtype, including float64; the framework's default
internal initialization otherwise creates float32 log variances.

## Mathematical problem and fitness

The implementation reuses `BurgersProblem`, `NormalizedFisiocomPINN` (wrapping
FisiocomPINN's `FullyConnectedNetwork` with Tanh activations), the residuals and
the loss registration in `pinn_model.py`.

Inputs `[N, 3]` contain physical `(t, x, y)` and outputs `[N, 2]` contain `(u, v)`.
The conservative equations are

```text
u_t + (u²/2)_x + (uv)_y = nu * (u_xx + u_yy)
v_t + (uv)_x + (v²/2)_y = nu * (v_xx + v_yy).
```

Domain, grid spacings, viscosity and initial-field constants come directly from
`control_dicts/mesh_properties.json` and `constant_properties.json`, or the
directory passed with `--config-directory`. With
`theta_x = 2*pi*(x-x_min)/(x_max-x_min)` and similarly for `theta_y`, the initial
fields are `u = mean_u + amplitude*sin(theta_x)*cos(theta_y)` and
`v = mean_v - amplitude*cos(theta_x)*sin(theta_y)`. All four spatial faces impose
homogeneous normal derivatives for both outputs. No physical parameters are
inferred and no observational/FVM data enters training. The physical formulation
and units are unchanged; only network inputs are scaled to `[-1, 1]`. Derivatives
are with respect to physical coordinates through that scaling.

Each DE genome contains every trainable weight and bias in
`model.named_parameters()` order, followed by log variances in the order
`s_Initial, s_Boundary, s_PDE`. Architecture, physical constants, normalization
buffers, and collocation points stay fixed. Metadata records the complete layout.
The initial network's log genes are zero, and the other individuals sample around
them according to the selected population initialization. `--log-weight-init-bound 4`
sets the log-gene initialization range; it does not constrain subsequent evolution.
Both DE and adaptive Adam allow unbounded log variances. Each positive loss
multiplier is `exp(-s_i)`.

The default DE objective is

```text
MSE_PDE = 0.5*(MSE_PDE_u + MSE_PDE_v).
fitness = exp(-s_Initial)*MSE_initial + s_Initial
        + exp(-s_Boundary)*MSE_boundary + s_Boundary
        + exp(-s_PDE)*MSE_PDE + s_PDE.
fixed_reference = MSE_initial + MSE_boundary + MSE_PDE.
```

This is exactly the local FisiocomPINN `AdaptiveLossWeights` formula used by
`train_pinn.py --adaptive`, including the grouping and averaging of both PDE
components into one loss. Adaptive weighting ignores the fixed `10:1:1`
coefficients, as does the ordinary adaptive PINN. At zero log variances, the
objective is the unweighted sum of the three MSEs.
For a fixed positive residual loss `L`, `exp(-s)*L+s` is minimized at
`s=log(L)`. The additive term penalizes suppressing a positive loss by making
its coefficient tend to zero. As in the framework, the objective is unbounded
below for an exactly zero component as its log variance tends to minus infinity;
finite precision can also overflow. Nonfinite DE objectives are rejected and
the raw losses and weights are retained for diagnosis. The genes are
optimization controls, not inferred physical uncertainties. This weighting
change does not alter the governing equations or physical conditions.

The initial, boundary and combined PDE terms are the same three FisiocomPINN
`LOSS` objects used by the ordinary PINN. The PDE uses `setBatchGenerator` and
`setEvalFunction` with residuals and targets of shape `[N, 2]`, ordered `(u, v)`;
MSE averages over both samples and components. The initial and boundary terms
also have two output components.

**Trainer adaptive weighting is enabled by default in both methods.** After DE,
the selected three log variances are copied into a real framework
`AdaptiveLossWeights` module, on the model's device and dtype, and included in
Adam's parameter groups. Adam jointly optimizes them and the network through
`Trainer` using the full regularized objective. No weights are frozen and no
constant objective offset is removed. DE and Adam differ in their update rules,
and the hybrid begins Adam from the DE-selected network and log variances.

Regularized fitness can be negative and can improve through weight adjustment
alone. Every generation therefore records the raw losses, selected coefficients,
fixed reference at the selected genome, and best fixed reference among all
evaluated candidates. The latter can belong to a rejected trial; its genome is
preserved separately. Neither reference replaces the selection objective. The
selected reference need not decrease monotonically. Held-out residuals and FVM
discrepancies remain separate validation metrics.

Use `--no-evolve-loss-weights` for the earlier fixed objective
`10*MSE_initial + MSE_boundary + MSE_PDE`. Coupled with the Adam runner's
`--no-adaptive`, this matches fixed loss weighting and disables adaptation in
both hybrid stages. Its fixed reference retains the same `10:1:1` coefficients.
Cached parent fitness remains valid
because each candidate carries its own genes and the objective formula and
deterministic points do not change across evaluations.

The old `--evolve-pde-weights` / `--no-evolve-pde-weights` flags are deprecated
aliases that emit a message explaining the new three-term behavior. The old
`--log-weight-bound` option is rejected explicitly: use `--log-weight-init-bound`
to control initialization only. Existing result directories are not rewritten;
the prior two-PDE-weight scheme is identified separately in comparison reports.

The spatial points are every FVM cell center. PDE times are all mesh time levels
after the initial instant; initial targets use the initial time; boundary points
use every time level on each face. There is no random sampling or subsampling.
Generators recreate the same coordinates as fresh autograd leaves on each call.
`--batch-size` controls only comparison inference, not training collocation.

## Evolution and transition

The mutation strategy is **DE/rand/1/bin**, as in the work_3 reference. The
default now uses dimension-independent, jDE-style parameter proposals:

- Three distinct donors exclude the target; mutation is
  `x_r1 + F_trial * (x_r2 - x_r3)`. Network coordinates are clipped to their
  bounds; adaptive log-variance genes are unbounded after initialization.
- Binomial crossover uses `CR_trial` and forces one mutant coordinate.
- Generations are synchronous. Strict `<` replacement preserves a target on
  ties, providing implicit elitism. All donors use the preceding population.
- Each individual starts with `F` uniform in `[0.3, 0.9]` and `CR` uniform in
  `[0, 1]`. Before each trial, each control independently has probability 0.1
  (`--adaptation-probability`) of being resampled within its range. The proposed
  controls generate that trial and survive only with strict improvement.
  This follows the proposal/selection mechanism of
  [Brest et al. (2006)](https://doi.org/10.1109/TEVC.2006.872133), while retaining
  this experiment's `[0.3, 0.9]` mutation-factor range.
- `--control-adaptation legacy` retains the original all-coordinate crossover
  gate and `[0.9, 1]` CR range. Its probability is `CR**(D-1)`, which makes
  adaptation extremely rare for the previous five-hidden-layer network's 4,418
  parameters (and still rare for the selected three-hidden-layer network). Copying
  that gate from a low-dimensional example made it ineffective here.
- One individual contains the unchanged Xavier-uniform/zero-bias initial
  network. With default `--initialization local`, other genes are uniform in
  `initial +/- initial_spread*(upper-lower)` and clipped to their bounds;
  `--initial-spread` defaults to 0.05. Thus network perturbations default to
  `[-0.2, 0.2]`, and log-variance perturbations to `[-0.4, 0.4]`.
  `--initialization uniform` restores full-range sampling, with log genes sampled
  in `[-log_weight_init_bound, log_weight_init_bound]`. The finite log-gene range
  is used only to initialize the population. Network bounds are a search
  configuration, not physical restrictions. They must contain the initial
  network; initialization is never silently clipped. Adam is unconstrained by
  the network's DE search bounds.
- Unlike `work_3`'s arbitrary evaluation-budget stopping, this schedule stops
  after a fixed number of complete generations and has no partial generation.
- Nonfinite fitness is counted and ranked as positive infinity. A population
  with no finite initial candidate fails explicitly. Programming errors propagate.

`--seed` controls network initialization. A separate local Torch generator,
seeded by `--de-seed`, controls the DE population and operators. There is no inner
stochastic PINN training per candidate and no collocation randomness. Identical
seeds are reproducible on the same tested CPU setup; bitwise agreement across
devices/software versions is not assumed. Run multiple independent seed pairs
before drawing conclusions about optimization performance.

DE evaluations copy genomes into the network without replacing Parameter
objects. The final evaluated trial may lose, so the returned global best is
explicitly copied into the model before Adam is constructed. Adam starts with
fresh moments and uses the registered losses and the best genome's log variances
as its initial adaptive state.

To reproduce the previous algorithm configuration explicitly, pass
`--no-evolve-loss-weights --control-adaptation legacy --initialization uniform`.
These options preserve the old operators and objective; they do not make the
old high-dimensional initialization well scaled.

The local Trainer's custom optimizer interface accepts a PyTorch Optimizer, but
its loop does not pass a reevaluation closure to `step()`. DE therefore uses an
isolated vector search calling the framework losses directly; Adam is passed as
an external optimizer to `Trainer`. No Trainer internals are patched.

## Artifacts and independent evaluation

Every invocation creates a new timestamped output directory, preserving earlier
results. `--output-directory` changes its parent directory.

| File | Contents |
| --- | --- |
| `metadata.json` | Configuration, seeds, genome layout, source hashes/framework commit, timings, counts, metrics and completion/failure status |
| `de_history.csv`, `de_history.npz` | Generation 0 through G: fitness statistics, raw losses, coefficients/log genes, fixed-reference diagnostics, accepted trials, control proposals/acceptances, normalized population spread and mean F/CR; CSV flushed every generation |
| `de_population.pt` | Final population, fitness, F/CR, best and initial genomes, initialization bounds, `bound_mask` identifying clipped network coordinates, and best fixed-reference genome among all evaluations |
| `de_best_model.pt` | Model state immediately before Adam |
| `pinn_model.pt`, `adam_optimizer.pt` | Final model and Adam optimizer state |
| `adaptive_weights.pt` | Final framework adaptive-weight state, when adaptation is enabled |
| `adam_loss_weights.npz` | `loss_names = [Initial, Boundary, PDE]`, and `log_vars`/`weights` arrays of shape `[Adam epochs, 3]`, recorded after each update |
| `losses.npz` | Individual unweighted losses **before each Adam update** |
| `comparison.npz` | Final PINN and FVM fields and per-time RMSE |

An independent PDE check after each stage uses quarter-cell offsets, covering up
to 21 evenly selected cells on each axis throughout the domain. These points are
never used by either optimizer. Final fields are compared with the unchanged
serial FVM on the same physical/grid configuration, reporting RMSE, MAE, maximum
absolute error and relative L2 error. FVM is an approximate numerical reference;
its cell values are compared with PINN point predictions at cell centers, and
these centers overlap the training spatial grid. The existing stability check
can reject an unstable FVM configuration; completed training artifacts remain
available and metadata records the failure.

Plot the stored comparison using the existing script:

```bash
python work_4/plot_pinn_results.py --run-directory work_4/pinn_sim/RUN
```

To compare **FVM, PINN Adam and PINN DE-to-Adam together**, repeat the run option:

```bash
python work_4/plot_pinn_results.py \
  --run-directory work_4/pinn_sim/ADAM_RUN \
  --run-directory work_4/pinn_sim/DE_ADAM_RUN
```

Without `--run-directory`, the comparator selects the latest completed saved
comparison for each optimizer in `pinn_sim` (or `--runs-directory`). If those
runs use different physical configurations, select compatible runs explicitly.
The comparator verifies the physical metadata, coordinates, time levels and
FVM fields before combining results. A single run remains supported, including
older artifacts without optimizer metadata, labeled as unknown.

Combined PNG panels and GIFs use shared color scales across methods and times,
including a shared absolute-error scale. RMSE curves distinguish methods and
velocity components. `comparison_metrics.csv` and `comparison_metrics.json`
report RMSE, MAE, maximum error, relative L2 error and per-run configuration,
including adaptive weighting and the distinct DE/Adam budgets. Adam loss curves
use Adam iterations; separate `de_convergence_RUN.png` figures use DE fitness
evaluations. They are not concatenated into an equal-cost epoch axis.
Regularized DE fitness uses a symmetric-log axis when nonpositive. Additional
`de_diagnostics_RUN.png` figures show the fixed reference and evolved coefficients.
`adam_loss_weights_RUN.png` shows the three coefficients after each Adam update
when the saved run includes `adam_loss_weights.npz`.
Reports distinguish current three-term adaptation from saved runs of the older
PDE-only variant; older artifacts remain readable and are not reinterpreted as
using the current formula.
Multi-run outputs default to a new `plots/pinn_comparison_TIMESTAMP` directory;
`--output-directory` overrides it. The notebook also provides an optional
`RUN_SAVED_PINN_COMPARISON` block for invoking this same comparator.

The inherited sinusoidal initial condition does not satisfy all zero-gradient
face conditions at the initial instant. This existing modeling limitation is
preserved and should be considered when interpreting accuracy. A smaller
training loss alone does not establish superiority over Adam-only training.

## Verification

The focused CPU tests exercise DE reproducibility and budgets, nonfinite
fitness handling, conservative residuals against manufactured polynomials,
float32/float64 loss derivatives, genome copying, and the exact best-to-Adam
handoff with finite gradients and actual optimizer updates:

```bash
python -m unittest discover -s work_4/tests -p 'test_*.py' -v
```

Tests cover float32/float64 residuals and genome handling, dimension-independent
control adaptation, loss grouping and regularized-objective equivalence with
the framework, and inheritance followed by Adam adaptation of all three weights.
Plot checks cover current weight names, legacy artifacts and negative regularized
fitness. Smoke tests verify execution on a deliberately small problem; their
loss values are not scientific evidence of better optimization.

Verification of the aligned scheme passed **48 tests** on CPU with PyTorch
`2.5.1` and local Pinn-Torch commit `ee4a05e01c4374d36ad19a3a6cc7266d8d441620`.
For identical network/log-weight states, the ordinary adaptive PINN and the
DE-to-Adam handoff produced exactly matching network and log-weight updates
across two Adam iterations in float32 and float64. End-to-end float64 smoke
checks exercised adaptive DE/Adam, legacy fixed DE/Adam, and adaptive Adam alone
on a 3-by-3 spatial grid with three time levels. Each hybrid used four individuals,
two DE generations (12 evaluations) and two Adam iterations. All completed
artifact serialization, independent residual checks and FVM comparison with
finite outputs. A regression test also injects coefficient overflow after Adam
and verifies that the failure remains recorded in valid JSON metadata.
No production-grid experiment was run for this change.

A separate [stagnation diagnostic](diagnostics/de_stagnation_20260909/README.md)
preserves eight reduced runs with the original fixed objective, source snapshots
and full raw histories. It reproduces flat best fitness with uniform initialization
and inactive legacy control adaptation. These are implementation diagnostics;
GPU execution, production-grid training and full independent-run scientific
comparisons have not been performed for the revised implementation.

A quick CLI smoke test can use a separate configuration directory with a coarse
grid and short time domain, passed explicitly via `--config-directory`, together
with `--epochs 4 --de-generations 2 --population-size 4 --hidden-sizes 4
--device cpu`. Such a deliberately reduced problem verifies execution only;
it is not a final scientific experiment. The repository control files are never
rewritten by this runner. Full experiments and multi-seed comparisons must be
run separately before making performance claims.
