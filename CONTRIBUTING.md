# Contributing to Evolutionary Algorithms

Thank you for your interest in contributing to this repository. Contributions
to the evolutionary computation implementations, EVRP experiments,
documentation, tests, and analyses are welcome. Because this is a research
repository, contributions should prioritize correctness, reproducibility, and
clear reporting of experimental conditions.

## Table of Contents

- [Reporting Issues](#reporting-issues)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Coding Standards](#coding-standards)
- [Scientific Reproducibility](#scientific-reproducibility)
- [Testing Changes](#testing-changes)
- [Commit Messages](#commit-messages)
- [Documentation and Results](#documentation-and-results)
- [License](#license)
- [Code of Conduct](#code-of-conduct)
- [Getting Help](#getting-help)

## Reporting Issues

Before opening an issue, search the
[existing issues](https://github.com/Esterci/Evolutionary_Algorithms/issues) to
check whether the problem or feature request has already been reported. Include
enough information for another contributor to reproduce and investigate the
issue:

- **Title:** Provide a clear and concise summary.
- **Description:** Explain the problem, proposed improvement, or research
  question.
- **Location:** Identify the relevant work directory, source file, function,
  notebook cell, or problem instance.
- **Steps to reproduce:** Include the exact build and execution commands.
- **Expected and actual behavior:** Describe what should happen and what was
  observed.
- **Experimental configuration:** Report the random seed, problem instance,
  population size, operators, rates, stopping condition, and number of fitness
  evaluations when applicable.
- **Logs and output:** Include concise compiler messages, tracebacks, or result
  excerpts. Do not include credentials or sensitive information.

## Submitting Pull Requests

1. Fork the repository on GitHub.

2. Clone your fork:

   ```bash
   git clone https://github.com/YOUR_USERNAME/Evolutionary_Algorithms.git
   cd Evolutionary_Algorithms
   ```

3. Create a focused branch:

   ```bash
   git checkout -b feature/short-description
   ```

4. Make the smallest reasonable set of changes for the proposed contribution.
   Preserve the existing organization of `work_1`, `work_2`, and `work_3`.

5. Build or run the affected method and perform the relevant checks described
   below.

6. Update documentation when commands, parameters, output formats, or
   algorithmic behavior change.

7. Commit and push your branch:

   ```bash
   git push origin feature/short-description
   ```

8. Open a pull request against the main repository. Explain the motivation,
   implementation, verification performed, and any limitations. Reference
   related issues where applicable.

Pull requests should address one coherent change. Avoid combining unrelated
refactoring, formatting, algorithm modifications, and experimental results.

## Coding Standards

### C and C++

- Follow the architecture and naming conventions already used in the relevant
  work directory.
- Compile without introducing new warnings under the flags in the existing
  `makefile`.
- Check array bounds, indexes, pointer lifetimes, allocation, deallocation, and
  uninitialized values carefully.
- Preserve EVRP feasibility: every customer must be served exactly as intended,
  vehicle capacity and battery constraints must hold, charging-station visits
  must be feasible, and each route must be able to return to the depot.

### Python and notebooks

- Follow the existing straightforward, function-based style.
- Keep scripts reproducible through explicit random seeds.
- Keep notebooks readable and ensure cells can be executed in order from a
  fresh kernel.
- Avoid adding plotting or analysis dependencies to command-line optimization
  scripts unless they are required by the method.

### Comments

- Write source-code comments in English.
- Explain why a non-obvious decision is necessary instead of restating the
  code.

## Scientific Reproducibility

Evolutionary algorithms are stochastic. Do not claim that a method is better or
that a bug is fixed based only on one favorable execution. When changing an
algorithm or presenting results:

- preserve or document random seeds;
- identify the problem instance and objective function;
- report population size, crossover rate, mutation rate or differential
  weight, selection pressure, and operators;
- report the stopping criterion and number of fitness evaluations;
- use multiple independent runs for performance comparisons;
- report both means and standard deviations for aggregated results;
- do not change parameters solely to improve reported outcomes; and
- distinguish implementation defects from ordinary stochastic behavior.

Do not fabricate, manually alter, or selectively omit experimental results.

## Testing Changes

Build the affected C++ implementation from its own directory because instance
and result paths are relative to the working directory:

```bash
cd work_1
make
```

or:

```bash
cd work_2
make
```

Run at least one relevant problem instance after compiling. For changes to the
Python DE method, run a small deterministic check and the documented reference
configuration, for example:

```bash
python3 work_3/rastrigin/de_best_1_bin_adpt.py
python3 work_3/rastrigin/plot_results.py
```

When changing a notebook, restart its kernel and run all cells in order. Review
the generated values and plots rather than only checking that execution
finishes.

> **Warning:** The current `make clean` and `make rebuild` targets also delete
> `stats/*.txt` and `evolution_curves/*.csv` in their respective work
> directories. Back up required experimental results before using these
> targets.

## Commit Messages

Write concise, descriptive commit messages. Use an imperative summary and add a
body when the motivation, algorithmic consequences, or verification requires
more explanation. Reference related issues when appropriate.

Example:

```text
fix: preserve feasible EVRP routes after mutation

Reject repaired routes that exceed vehicle capacity and document the
validation performed on E-n22-k4.evrp.

Closes #42
```

Common prefixes such as `feat`, `fix`, `docs`, `test`, and `refactor` are
welcome but not mandatory.

## Documentation and Results

- Update `README.md` when repository organization, requirements, commands, or
  result locations change.
- Document new command-line parameters and their valid ranges.
- Include units, column meanings, and aggregation procedures for new result
  formats.
- Do not commit large generated outputs unless they are required to reproduce a
  documented experiment.
- Never commit credentials, API keys, passwords, private keys, or local
  environment files.

## License

By contributing to this repository, you agree that your contributions will be
licensed under the GNU General Public License, version 3 (`GPL-3.0-only`). See
the [LICENSE](LICENSE) file for the complete license text.

## Code of Conduct

Treat other contributors with respect. Keep technical discussions constructive,
welcome questions from people with different levels of experience, and focus
reviews on the work rather than the person. Harassment, discrimination, and
personal attacks are not acceptable.

## Getting Help

If you need clarification about the code, experiments, or contribution process,
open an issue in the
[GitHub issue tracker](https://github.com/Esterci/Evolutionary_Algorithms/issues).

Thank you for helping improve the Evolutionary Algorithms repository.
