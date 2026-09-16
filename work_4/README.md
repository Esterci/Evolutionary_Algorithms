# Work 4: Burgers experiments

Run all commands below from `work_4`. This README is the only file at this
level; implementations, entry points, documentation, and artifacts live in
subdirectories.

```text
work_4/
├── README.md
├── methods/        # FVM solver, PINN model, differential evolution, DE/Adam
├── runners/        # FVM, PINN training, and paired-study entry points
├── utils/          # Plotting and study-analysis tools
├── shells/         # Paired-experiment launchers and result cleanup
├── docs/           # Detailed DE/Adam and cleanup documentation
├── notebooks/      # Exploratory numerical notebook
├── tests/          # Automated checks and small training smoke tests
├── control_dicts/  # Mesh and physical configuration JSON files
├── fvm_sim/        # Stored FVM solutions and metadata
├── pinn_sim/       # Trained PINNs, histories, and comparisons
├── plots/          # Generated figures and animations
├── studies/        # Recorded studies, reports, and source snapshots
└── diagnostics/    # Recorded diagnostics and source snapshots
```

## Execution

Activate a Python environment with PyTorch, NumPy, Matplotlib, Pillow, and
FisiocomPINN. The existing sibling checkout `../../Pinn-Torch` is preferred
when present; otherwise the installed `fisiocomPinn` package is used.

```bash
cd work_4  # From the repository root; only needed once.
python -m runners.run_fvm --help
python -m runners.train_pinn --help
python -m runners.train_pinn_de_adam --help
python -m runners.run_paired_study --help
python -m utils.plot_results --help
python -m utils.plot_pinn_results --help
python -m utils.analyze_paired_study --help
python -m utils.analyze_de_budget_study --help
```

Remove `--help` and supply the desired experiment arguments to run a command.
Default configuration and output paths remain under `work_4`; explicitly
provided relative paths are resolved from the current working directory.
For example, plot a completed run with
`python -m utils.plot_pinn_results --run-directory pinn_sim/RUN`.

```bash
bash shells/run_recommended_pair.sh
bash shells/run_de32_pair.sh
bash shells/clean_results.sh fvm --dry-run
python -m unittest discover -s tests -p 'test_*.py' -v
```

The paired launchers run full training schedules. `PYTHON_BIN` selects their
Python interpreter. Cleanup uses Python 3; `--dry-run` only lists targets.

See [DE/Adam details](docs/README_de_adam.md) for experimental settings and
[cleanup details](docs/README_cleanup.md) for deletion behavior.
Existing results and historical source snapshots are preserved; paths recorded
inside historical artifacts describe the layout at the time of those runs.
Python may also create `__pycache__` directories during execution.

See [scheduling and early stopping](docs/README_training_controls.md) for the
new default training controls, required framework version, and actual budgets.
Set `FISIOCOMPINN_PATH` to select an updated framework checkout explicitly.

For the paired four-mode loss-coefficient experiment, use
`bash shells/run_loss_weight_study.sh`. See the
[experiment protocol and configurable shell variables](docs/README_loss_weight_study.md).
