# Evolutionary Algorithms

Coursework and experiments developed for a PhD course on Evolutionary
Algorithms. The repository currently contains evolutionary approaches for the
Electric Vehicle Routing Problem (EVRP) and a compact Differential Evolution
(DE) study on a two-dimensional numerical benchmark.

The EVRP programs are stochastic experiments. Each invocation performs 20
independent runs and stops each run after `25000 * ACTUAL_PROBLEM_SIZE` fitness
evaluations. Compare configurations using the complete set of runs rather than
a single route or execution.

## Repository organization

```text
.
├── work_1/                    Initial C++ EVRP evolutionary algorithm
│   ├── main.cpp               Experiment entry point
│   ├── heuristic.{cpp,hpp}    Evolutionary operators and route construction
│   ├── EVRP.{cpp,hpp}         Instance reader, objective, and validation
│   ├── stats.{cpp,hpp}        Statistics and evolution-curve output
│   ├── E-*.evrp               EVRP benchmark instances
│   ├── stats/                 Per-configuration results
│   └── evolution_curves/      Per-generation fitness summaries
├── work_2/                    Extended EVRP implementation and experiments
│   ├── main.cpp               Experiment entry point with optional verbosity
│   ├── heuristic.{cpp,hpp}    Operators, route construction, and repair
│   ├── EVRP.{cpp,hpp}         EVRP model, evaluation, and validation
│   ├── stats.{cpp,hpp}        Raw result and convergence output
│   ├── gridsearch.py           Parallel parameter-grid runner
│   ├── gridsearch.job          Example PBS batch script
│   ├── result_analysis.ipynb   Statistical analysis and plot generation
│   ├── stats/                  Raw experiment results
│   ├── evolution_curves/      Raw convergence data
│   ├── evolution_plots/       Generated convergence plots
│   ├── route_plots/           Generated route plots
│   └── tables/                Generated summary tables
└── work_3/
    ├── rastrigin/
        ├── de_best_1_bin_adpt.py  Evaluation-budget DE runner
        ├── plot_results.py         Independent-run convergence plots
        ├── result_analysis.ipynb  Statistical analysis notebook
        ├── evolution_curves/      Per-seed convergence data
        ├── evolution_plots/       Generated plots
        └── results/               Appended run-level results
    └── gearbox/
        ├── de_rand_1_bin_adpt.py  Constrained speed-reducer DE runner
        ├── plot_results.py         Fitness and constraint-violation plots
        ├── result_analysis.ipynb  Statistical analysis notebook
        ├── evolution_curves/      Per-seed convergence data
        ├── evolution_plots/       Generated plots
        └── results/               Appended run-level results
```

`work_1` and `work_2` are separate programs. Build and run each one from its
own directory because instance and output paths are relative to the current
working directory.

## Requirements

For the C++ EVRP programs:

- GNU Make
- A C++ compiler with C++11 support or newer (`g++` is used by the makefiles)

For the Python analysis and notebooks:

- Python 3
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib
- SciPy
- PyTorch (required by the `work_3` scripts and notebook)

One possible Python setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install jupyter numpy pandas matplotlib scipy torch
```

## Compile the EVRP methods

Build the initial implementation:

```bash
cd work_1
make
```

Build the extended implementation:

```bash
cd work_2
make
```

Both commands create an executable named `main` in the corresponding
directory. Use `make rebuild` to rebuild all objects. Be careful with `make
clean`: besides the executable and object files, the current makefiles also
remove `stats/*.txt` and `evolution_curves/*.csv`.

## Run the EVRP methods

### Work 1

From `work_1`, run:

```bash
./main INSTANCE MUTATION_RATE CROSSOVER_RATE MUTATION_METHOD SELECTION_PRESSURE POPULATION_SIZE
```

Example:

```bash
cd work_1
./main E-n22-k4.evrp 0.90 0.80 ins 1.50 32
```

### Work 2

`work_2` accepts the same parameters plus a verbosity flag:

```bash
./main INSTANCE MUTATION_RATE CROSSOVER_RATE MUTATION_METHOD SELECTION_PRESSURE POPULATION_SIZE VERBOSE
```

Example:

```bash
cd work_2
./main E-n22-k4.evrp 0.80 0.70 ins 1.50 32 1
```

Set `VERBOSE` to `1` to print the selected parameters and completion banner, or
to `0` for grid-search runs.

The positional parameters are:

| Parameter | Meaning |
| --- | --- |
| `INSTANCE` | EVRP instance file, such as `E-n22-k4.evrp` or `E-n76-k7.evrp` |
| `MUTATION_RATE` | Mutation probability |
| `CROSSOVER_RATE` | Crossover probability |
| `MUTATION_METHOD` | `swp` (swap), `ins` (insertion), `mix` (scramble), or `inv` (inversion) |
| `SELECTION_PRESSURE` | Selection-pressure parameter |
| `POPULATION_SIZE` | Number of individuals in the population |
| `VERBOSE` | `work_2` only: `0` for quiet output or `1` for verbose output |

The programs currently expect every positional argument shown above; invoking
them with an incomplete argument list is not supported.

### Parameter grid search

Edit the parameter arrays near the top of `work_2/gridsearch.py`, compile the
`work_2` executable, and run the script from `work_2`:

```bash
cd work_2
make
python3 gridsearch.py
```

The script constructs the Cartesian product of the configured parameters and
instances, then runs up to eight configurations concurrently. Adjust
`max_workers` to match the available CPU and memory. `gridsearch.job` is a
cluster-specific PBS example and contains an environment path that must be
adapted before use on another system.

## Run the notebooks

Start Jupyter from the repository root:

```bash
jupyter lab
```

Then open one of the following:

- `work_2/result_analysis.ipynb` reads the raw `stats/` and
  `evolution_curves/` files and creates statistical summaries, route plots,
  and convergence plots. Its relative paths assume that the notebook kernel
  runs with `work_2` as the working directory.
- `work_3/rastrigin/result_analysis.ipynb` summarizes the independent
  `DE/rand/1/bin` executions and their recorded convergence data.
- `work_3/gearbox/result_analysis.ipynb` summarizes the independent constrained
  speed-reducer executions, feasibility, and convergence data.

### Self-adaptive DE/rand/1/bin runner

Configure `POPULATION_SIZE`, `DIMENSION`, `MAX_FITNESS_EVALUATIONS`,
`EVALUATION_STEP`, and `NUMBER_OF_RUNS` at the beginning of the runner, then
execute it from `work_3/rastrigin`:

```bash
python3 de_best_1_bin_adpt.py
```

The script retains the sequential `DE/rand/1/bin` structure: mutation
`r1 + F * (r2 - r3)`, bound repair by
clamping, binomial crossover with one forced mutant dimension, and greedy
selection. Each individual carries self-adaptive `F` and `CR` values,
initialized in `[0.3, 0.9]` and `[0.9, 1.0]`. Accepted offspring inherit their
candidate control parameters. Execution uses the CPU and `float64`.

The initial population counts toward the budget. If the remaining budget is
smaller than the population size, that many targets are selected randomly,
without replacement, in the last partial generation. The default experiment
performs 20 independent runs,
using the run number as the seed (`1` through `20`). The function does not
return an experimental result. It writes one convergence CSV per seed to
`work_3/rastrigin/evolution_curves/`, with columns `seed`,
`fitness_evaluations`, `fit_mean`, and `fit_std`. Final results from all runs
of the same configuration are appended as rows to one CSV in
`work_3/rastrigin/results/`. Running the script again appends another 20 rows.

After generating the convergence CSVs, create the plots with:

```bash
cd work_3/rastrigin
python3 plot_results.py
```

The script identifies the execution with the lowest final objective value for
each experimental configuration. It plots that execution's population mean
fitness with a band of one population standard deviation and reports the best
seed and the final-population mean and standard deviation of adaptive `F` and
`CR`. All plot text is in Portuguese. This analysis does not compare different
algorithms or models.

### Constrained speed-reducer runner

The speed-reducer study uses self-adaptive `DE/rand/1/bin` with seven design
variables, 11 inequality constraints, and a quadratic exterior penalty. The
third design variable is rounded to an integer after initialization, mutation,
and crossover repair. Configure the population, evaluation budget, sampling
step, number of independent runs, feasibility tolerance, and penalty weight at
the beginning of the script, then run:

```bash
cd work_3/gearbox
python3 de_rand_1_bin_adpt.py
python3 plot_results.py
```

The default configuration uses population size 30, an exact budget of 36,000
objective-function evaluations, and seeds `1` and `2`. Each convergence CSV
records penalized-fitness statistics and the mean and population standard
deviation of every nonnegative constraint violation. Final summaries report
the objective, penalty, feasibility, maximum violation, feasible percentage,
adaptive parameters, and best design vector. Feasible solutions are ranked by
the unpenalized objective; penalized fitness is used only as a fallback when a
run has no feasible individual.

## Results

Each EVRP execution writes files whose names encode the experimental
configuration. For example:

```text
pi-E-n22-k4.evrp--mtr-0.800000--cor-0.700000--mtm-ins--stp-1.500000--pop-32.txt
```

The abbreviations are `pi` (problem instance), `mtr` (mutation rate), `cor`
(crossover rate), `mtm` (mutation method), `stp` (selection pressure), and
`pop` (population size).

Results are located at:

- `work_1/stats/*.txt`: routes, run-level objective values, mean, standard
  deviation, minimum, and maximum for the initial implementation.
- `work_1/evolution_curves/*.csv`: generation-level mean fitness and standard
  deviation for each seed/run.
- `work_2/stats/*.txt`: extended raw results, including route feasibility.
- `work_2/evolution_curves/*.csv`: convergence data with columns `seed`,
  `generation`, `fit_mean`, and `fit_std`.
- `work_2/tables/top_5_*.csv`: top configurations produced by the analysis
  notebook.
- `work_2/evolution_plots/*.png`: generated fitness-curve figures.
- `work_2/route_plots/*.png`: generated plots of the leading routes.
- `work_3/rastrigin/result_analysis.ipynb`: statistical summaries of the
  independent DE executions.
- `work_3/rastrigin/evolution_curves/*.csv`: population fitness mean and standard
  deviation sampled by objective-function evaluation count.
- `work_3/rastrigin/results/*.csv`: appended run-level configurations, seeds, best
  solutions, adaptive-parameter statistics, and convergence-curve paths.
- `work_3/rastrigin/evolution_plots/*.png`: convergence and population-dispersion
  figures for the best execution of each experimental configuration.
- `work_3/gearbox/result_analysis.ipynb`: statistical summaries of the
  independent constrained speed-reducer executions.
- `work_3/gearbox/evolution_curves/*.csv`: penalized-fitness and per-constraint
  violation statistics sampled by objective-function evaluation count.
- `work_3/gearbox/results/*.csv`: run-level objectives, penalties, feasibility,
  adaptive-parameter statistics, and design vectors.
- `work_3/gearbox/evolution_plots/*.png`: fitness and constraint-violation
  figures for the selected execution.

The EVRP statistics files are opened in append mode. Repeating exactly the
same configuration adds data to the existing files; archive or rename prior
outputs when a fresh experimental dataset is required.

## Contributors and citation

This repository is maintained with contributions from:

- Thiago Esterci Fernandes (also represented as `Esterci` and `Thiago` in the
  Git history)
- `aclaraverly`

The list is derived from the repository's Git history. See [AUTHORS](AUTHORS)
for the contributor record and [CITATION.cff](CITATION.cff) for
machine-readable citation metadata.

Suggested citation:

> T. E. Fernandes and aclaraverly, *Evolutionary Algorithms: EVRP and
> Differential Evolution Studies*, software, 2026. [Online]. Available:
> https://github.com/Esterci/Evolutionary_Algorithms

BibTeX example:

```bibtex
@software{fernandes_evolutionary_algorithms_2026,
  author  = {Fernandes, Thiago Esterci and aclaraverly},
  title   = {Evolutionary Algorithms: EVRP and Differential Evolution Studies},
  year    = {2026},
  url     = {https://github.com/Esterci/Evolutionary_Algorithms},
  license = {GPL-3.0-only}
}
```

## License

This project is free software distributed under the GNU General Public
License, version 3 (`GPL-3.0-only`). See [LICENSE](LICENSE) for the complete
license text and [NOTICE](NOTICE) for the copyright and warranty notice.
