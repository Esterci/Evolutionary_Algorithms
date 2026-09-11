# Cleaning generated results

Run from the repository root (Bash and Python 3 are required):

```bash
./work_4/clean_results.sh fvm --dry-run
./work_4/clean_results.sh fvm
./work_4/clean_results.sh pinn
./work_4/clean_results.sh pinn-de
```

Each command deletes only the selected method's results. Add `--dry-run` to
any command to list targets without removing them. Paths are relative to the
script, so invoking it from another working directory also works.

- `fvm`: generated solution arrays and metadata in `fvm_sim`, plus FVM velocity
  PNG/GIF files in `plots`.
- `pinn`: Adam runs identified by metadata in `pinn_sim`, plus the notebook's
  flat `pinn_model_*`, `pinn_losses_*`, and `pinn_metadata_*` artifacts.
- `pinn-de`: DE/Adam runs identified by metadata in `pinn_sim`.

Selected run directories are removed with all their contents, including local
plots and checkpoints. Unclassified runs, shared multi-run comparison plots,
and custom output directories are preserved. Stop writers before cleaning
their results. Deletion is permanent; cleanup does not run training or modify
the mathematical model or FisiocomPINN.
