# Four-mode DE/PINN loss-weight experiment

This experiment compares which stage updates the Initial, Boundary, and PDE
loss coefficients. All four pipelines train the network using DE followed by
Adam. It does not compare DE-only network training with Adam-only training.

| Mode | DE coefficients | Adam coefficients |
| --- | --- | --- |
| `de` | Evolved log variances | Frozen at the selected DE values |
| `pinn` | Fixed configured coefficients | Adaptive, starting from configured coefficients |
| `both` | Evolved log variances | Adaptive, starting from selected DE values |
| `none` | Fixed configured coefficients | Same fixed coefficients |

New runs start with Initial=1000, Boundary=1000 and combined PDE=1000.
Historical runs retain their recorded settings. No full experiment with these
new defaults has been run; reduced tests do not establish solution accuracy.

## Run from work_4

Activate the environment containing PyTorch, NumPy, Matplotlib, and FisiocomPINN.
An updated framework supporting scheduler and early stopping is required. If the
sibling checkout is old, select an updated checkout with `FISIOCOMPINN_PATH`.

```bash
cd work_4
conda activate torch-numba-11
# Only when the default sibling checkout is not the updated framework:
export FISIOCOMPINN_PATH=/path/to/updated/Pinn-Torch

GENERATIONS=100 RUNS=3 POPULATION_SIZE=32 \
  bash shells/run_loss_weight_study.sh
```

| Shell variable | Default | Meaning |
| --- | ---: | --- |
| `GENERATIONS` | 100 | DE generations in every pipeline |
| `BUDGET` | 8064 | Maximum total DE evaluations + Adam updates per run |
| `RUNS` | 3 | Independent seed pairs per pipeline; total runs = 4 × RUNS |
| `POPULATION_SIZE` | 32 | DE population size |
| `NETWORK_SEED` | 2028 | First network seed; incremented for each replicate |
| `DE_SEED` | 3028 | First DE seed; incremented for each replicate |
| `PYTHON_BIN` | python | Python interpreter |
| `DEVICE` | runner default | CPU or available CUDA device |
| `CONFIG_DIRECTORY` | control_dicts | Physical configuration directory |
| `OUTPUT_DIRECTORY` | new timestamped study | New output directory; existing directories are refused |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | 1 | CPU thread limits |

The shell runs from `work_4`, so supplied relative paths are relative to it.
Extra training options are forwarded unchanged to all four modes:

```bash
GENERATIONS=100 RUNS=5 \
  bash shells/run_loss_weight_study.sh \
  --hidden-sizes 32 32 32 --lr 0.001 --dtype float32 \
  --scheduler logarithmic --lr-final-ratio 0.1 \
  --early-stopping-patience 300 --early-stopping-tolerance 1e-5
```

Use `--scheduler none --no-early-stopping` for constant LR and a fixed number
of Adam updates. The scheduler starts at Adam, after DE. The study calculates
`Adam epochs = BUDGET - POPULATION_SIZE * (GENERATIONS + 1)` and converts this
to the existing runner's schedule-slot convention internally. The initial DE
population is included in the cost. With the default budget 8064 and population
32, 100 generations leave 4832 Adam epochs; 200 generations leave 1632.
Changing only `GENERATIONS` preserves the maximum total evaluation budget.
Configurations leaving fewer than one Adam update are rejected before training.
Use `BUDGET=5000` to choose a different total budget. The previous `EPOCHS`
variable and study `--epochs` option have been replaced by `BUDGET` / `--budget`.
Equal evaluation counts do not imply equal runtime: Adam also backpropagates. Actual updates and evaluations
are reported separately when early stopping triggers. Diagnostic evaluations
and FVM evaluation are outside this budget. See
[training controls](README_training_controls.md) for monitor and checkpoint semantics.

The shell exits nonzero if any run fails, but attempts the remaining pipelines
and generates the partial report. Failures and unstarted/interrupted records
remain explicit. There is no automatic retry or resume; a new invocation creates
a new study. Do not merge incomplete attempts without an explicit protocol.

## Scientific protocol

The Burgers equations, domain, Neumann boundaries, analytical initial condition,
normalization, and deterministic full-batch collocation remain those specified
by the copied configuration and existing methods. FisiocomPINN `LOSS`, `Trainer`,
`FullyConnectedNetwork`, and `AdaptiveLossWeights` remain the training components.
DE representation, selection, crossover, mutation, constraints, and stopping
remain those documented in [DE/Adam details](README_de_adam.md).

Each replicate uses the same network and DE seeds across modes; different
replicates use different seed pairs. Run order rotates by replicate. Different
DE genome dimensions consume random values differently: paired seeds do not
imply identical populations or mutations across all modes. No FVM targets enter
DE fitness, Adam training, or early stopping.

A shared float64 donor-cell FVM reference uses `h/2` and `k/2`. This changes only
the evaluation grid. Validation evaluates refined time indices 1,5,9,...;
test evaluates 3,7,11,... . Both use refined spatial centers and times offset
from training. The time sets are disjoint and are not used for tuning by this
runner. Require at least two original time intervals to obtain both splits.
The final models are evaluated at all these points; no best-seed selection occurs.

Joint RMSE is `sqrt(mean((prediction - reference)^2))` across points and the two
velocity components. Independent PDE RMSE uses the existing quarter-cell-offset
physics evaluation. Initial RMSE uses the refined initial grid. FVM agreement
is not exact-solution accuracy; cell-average/pointwise comparison and finite
reference resolution remain limitations. No statistical superiority is assumed.

## Artifacts and analysis

```text
studies/loss_weight_study_TIMESTAMP/
├── manifest.json              # Configuration, seeds, budgets, source hashes
├── records.json               # Every planned run, command, status, error, metrics
├── config/                    # Copied physical JSON inputs
├── logs/                      # One combined stdout/stderr log per run
├── runs/seed_SEED_MODE/RUN/    # Model, optimizer, scheduler, losses, evaluations
├── refined_fvm.npz            # Shared numerical reference
├── refined_fvm_metadata.json
└── analysis/
    ├── README.md              # Generated report and figures
    ├── runs.csv / runs.json   # Individual results including failures
    ├── summary.csv / summary.json
    ├── paired_differences.csv # Within-seed left-minus-right metric differences
    ├── comparison.png         # Paired metrics plus boundary MSE and training time
    ├── test_error_by_time.png # Mean and sample SD by physical time
    ├── training_losses.png   # Fixed vs adapted weighted MSE, best validation run per mode
    ├── de_convergence.png     # Mean best and population-mean DE fitness
    ├── de_diagnostics.png     # DE reference/weights above; mean F and CR below
    ├── pinn_loss_weights.png  # Best-run coefficients during Adam (de, pinn, both)
    ├── pinn_fixed_reference.png # Best-run fixed-reference losses during Adam
    ├── mean_histories.csv     # Plotted means, sample SD, counts and seeds
    └── fields_seed_*.png      # First planned seed, initial/middle/final fields
```

The report is generated automatically. Regenerate it without PyTorch or training:

```bash
python -m utils.analyze_loss_weight_study studies/loss_weight_study_TIMESTAMP
```

Only `analysis/` is rewritten. Raw artifacts are preserved. Summary statistics
include count, mean, sample standard deviation, median, minimum, maximum, and
IQR. Standard deviation is unavailable for a single run. Incomplete counts are
reported per mode; only completed finite results enter statistics. Pairwise
CSV rows require both modes to complete for that seed. No significance test or
confidence interval is claimed. Early-stopped curves are neither padded nor
averaged into survivor-biased tails.

Representative field figures use the first planned seed, not the best result,
and the original training-grid FVM comparison already saved by the trainer.
Their reference resolution differs from the refined-grid evaluation tables.
Missing representative modes are omitted explicitly, not replaced with other seeds.

## Small smoke test

Use a separate tiny configuration (for example h=0.5, k=0.01, t=[0,0.02] with
unchanged spatial domain and physical constants) in a temporary directory:

```bash
CONFIG_DIRECTORY=/path/to/tiny/config DEVICE=cpu \
GENERATIONS=1 BUDGET=10 RUNS=1 POPULATION_SIZE=4 \
  bash shells/run_loss_weight_study.sh --hidden-sizes 4 --dtype float64
```

A smoke test checks the pipeline and artifacts; it is not an accuracy result.

Curve means use the common history prefix within each mode, keeping the same
seed cohort throughout. Fields are separated into u and v figures (the `_v`
suffix identifies v); each has predictions in the first column and errors in the second, labeling both network
and DE seeds. Weight analysis concerns loss coefficients, not network parameters.

`training_losses.png` pairs a common fixed 1:1:1 weighting (left) with the
actual adapted coefficient times each MSE (right). The de coefficients were
adapted by DE and frozen during Adam; pinn and both adapt during Adam; none
retains its configured fixed coefficients. Initial coefficients and the previous
post-update snapshot align weights with the pre-update losses. Each product
is computed per seed before aggregation. Additive log-variance regularization
is excluded: these panels show weighted MSE contributions, not the full objective.

The second row of `de_diagnostics.png` compares de and both using the saved
population means of F and CR by generation, averaged across seeds. Shading
shows sample SD across those seed means; generation zero is initialization.

## Hidden architecture and DE adaptation

The shell forwards the following options unchanged to every mode (`de`, `pinn`,
`both`, `none`), preserving the same architecture for the comparison:

```bash
bash shells/run_loss_weight_study.sh \
  --hidden-sizes 32 32 32 \
  --activation-functions Tanh ReLu Tanh \
  --adaptation-probability 0.2
```

Supply one activation per hidden layer; omission selects Tanh for every layer.
Names are case-insensitive (`ReLu` is recorded as `ReLU`). Supported framework
activations are Tanh, ReLU, Elu, LeakyReLU, Sigmoid, Softplus, SELU, CELU, GELU,
and SiLU. RReLU is excluded because it is stochastic; GLU changes dimensions;
Linear requires layer dimensions. The output remains linear, with three inputs
(t, x, y) and two outputs (u, v). Hidden sizes and canonical activation names
are saved in metadata and used when loading models for evaluation. Old metadata
without activation names defaults to Tanh.

The local subclass extends FisiocomPINN layer construction because its standard
network fixes Tanh and its configurable variant also applies Tanh to the output.
The framework itself is unchanged. Xavier initialization is retained for all
activations. ReLU is allowed as requested; its second derivative is zero almost
everywhere, which matters for the viscous Burgers residual. Smooth activations
remain the default. Changing activations is an experimental configuration change.

`--adaptation-probability` defaults to 0.1 and must lie in [0, 1]. In jDE it
controls independent proposals for new F and CR, not their numerical ranges.
This study always uses DE followed by Adam; `pinn` names the stage adapting loss
coefficients, not a network trained exclusively with Adam. The standalone
`python -m runners.train_pinn` also supports the same architecture options.

## Shared initial loss coefficients

Edit `DEFAULT_LOSS_WEIGHTS` in `methods/loss_weight_config.py` to set defaults
for every runner, or override them without editing code:

```bash
bash shells/run_loss_weight_study.sh --initial-loss-weights 1000 1000 1000
```

The three values are Initial, Boundary and combined PDE. The standalone PINN
accepts the same flag; Python callers can pass `initial_loss_weights` as a named
dictionary to `build_trainer`. Split PDE terms each receive half the combined
coefficient: PDE_u=PDE_v=500 by default. DE-adaptive training still requires a
combined PDE term.

Adaptive coefficients use s=-log(w), giving s≈-6.907755 for w=1000 and
s≈-6.214608 for w=500. The first DE individual preserves these coefficients.
Other individuals use the existing uniform/local population sampling in a box
centered on the initial s values, with half-width `--log-weight-init-bound`
(default 4). This replaces the historical box centered on zero; local sampling
still uses `--initial-spread` times the full box width. F/CR initialization,
selection, mutation and crossover are unchanged. No loss-gene clipping occurs
after initialization. The exact per-term box is recorded in metadata.

The `de` and `both` modes pass selected DE log variances to Adam without a reset;
`pinn` starts Adam from the configured coefficients; `none` keeps them fixed.
The adaptive regularizer sum(s_i) remains in the objective. DE's separate
reference diagnostic remains the unweighted MSE sum when loss genes evolve;
fixed-weight modes use their registered coefficients. Analysis may additionally
use its explicitly labeled common 1:1:1 reference.

## Best-run loss plots and boundary comparison

PINN loss/reference/coefficient plots select one completed run per mode using
minimum validation RMSE (ties use the smaller seed). The same run supplies every
loss term, and legends identify network and DE seeds. These plots show selected
trajectories without seed averaging or SD bands. `selected_pinn_runs.json`
records the selection criterion, paths and scores. DE diagnostics and comparison
statistics still aggregate all completed runs. `mean_histories.csv` distinguishes
best-run trajectories from seed means in its `aggregation` column.

`comparison.png` places final raw boundary training MSE in the fifth panel and
training time in the sixth. The boundary value comes from `final_losses.Boundary`
in each run metadata, not an independent validation grid. Evaluation counts
remain in the tabular results.
