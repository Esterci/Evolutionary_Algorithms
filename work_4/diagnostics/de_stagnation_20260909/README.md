# Reduced DE stagnation diagnostic

These stored runs investigate implementation behavior, not solution accuracy or
optimizer superiority. `diagnostic.json` preserves every evaluated fitness,
generation history, seed, configuration, software version, and source hashes.
The accompanying source snapshots are the exact files used for those runs.
They precede the later addition of the history-only `best_index` field.

The problem uses CPU float64, one Torch thread, a 3-by-3 spatial grid and two
time intervals, a `[32,32,32,32,32]` network (4,418 parameters), population 12,
eight generations, and DE seed 2027. Each run receives 108 fitness evaluations.
All four combinations of local/uniform initialization and legacy/jDE-style
adaptation were checked for each of two network seeds. The fitness remains
`10*MSE_initial + MSE_boundary + MSE_PDE`: loss weights do not evolve in this
diagnostic, isolating the initialization/control changes.

| Network seed | Initialization / adaptation | Generation-zero best | Generation-eight best |
| --- | --- | ---: | ---: |
| 2026 | Uniform / legacy | 252.3908 | 252.3908 |
| 2026 | Local / jDE-style | 252.3908 | 170.6154 |
| 2028 | Uniform / legacy | 673.6985 | 673.6985 |
| 2028 | Local / jDE-style | 288.8409 | 71.0348 |

The initial best differs for seed 2028 because local initialization already
finds a candidate better than the preserved Xavier network. Full intermediate
configurations and unrounded values are in the JSON; no run was discarded.
Uniform runs accept candidates and lower population mean fitness while the
best remains flat. Their original F/CR adaptation never activates. This
distinguishes poor scaling/ineffective control adaptation from broken selection.

`source_0_pinn_de_audit_script.py` reproduces the diagnostic against the current
checkout and writes a new directory under `/tmp`. Its recorded ROOT is the
workspace used for this audit. Source snapshots preserve the original versions
for traceability; the script imports the live checkout, not these snapshots.

The evolved PDE-weight objective is validated separately by the mathematical,
Adam-handoff and end-to-end tests in `work_4/tests`. These eight short fixed-loss
runs are insufficient for statistical claims or production-grid conclusions.
