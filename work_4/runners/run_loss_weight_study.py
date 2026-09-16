"""Run four paired loss-weight adaptation pipelines and preserve every attempt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from methods.training_controls import require_training_control_api
from runners.train_pinn_de_adam import parse_args as training_args, source_metadata
from runners.run_paired_study import reference_arrays, evaluate, write_json

BASE = Path(__file__).resolve().parents[1]
MODES = ('de', 'pinn', 'both', 'none')


def adam_epochs_for_budget(budget, population_size, generations):
    """Reserve one evaluation per initial individual and per DE trial."""
    if budget < 1 or population_size < 4 or generations < 1:
        raise ValueError('Require budget >= 1, population-size >= 4 and generations >= 1')
    epochs = budget - population_size * (generations + 1)
    if epochs < 1:
        raise ValueError('Budget must exceed population-size * (generations + 1), leaving at least one Adam update')
    return epochs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=('Shared training options are forwarded to all four modes, including '
                '--hidden-sizes 32 32 32 --activation-functions Tanh ReLu Tanh '
                '--adaptation-probability 0.2. Supply one activation per hidden layer; '
                'defaults are Tanh and adaptation probability 0.1. Initial loss coefficients '
                'can be set with --initial-loss-weights 1000 1000 1000.'))
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--budget', type=int, default=8064, help='Maximum total DE fitness evaluations + Adam updates per run')
    parser.add_argument('--runs', type=int, default=3, help='Independent seeds per pipeline')
    parser.add_argument('--population-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2028)
    parser.add_argument('--de-seed', type=int, default=3028)
    parser.add_argument('--config-directory', type=Path, default=BASE / 'control_dicts')
    parser.add_argument('--output-directory', type=Path,
                        default=BASE / 'studies' / f'loss_weight_study_{time.time_ns()}')
    args, extra = parser.parse_known_args(argv)
    if args.runs < 1:
        parser.error('runs must be positive')
    try:
        adam_epochs = adam_epochs_for_budget(args.budget, args.population_size, args.generations)
    except ValueError as error:
        parser.error(str(error))
    # Prevent additional training options from overriding study pairing or paths.
    reserved = {'--epochs', '--de-generations', '--population-size', '--seed', '--de-seed',
                '--output-directory', '--config-directory', '--loss-weight-stage',
                '--evolve-loss-weights', '--no-evolve-loss-weights',
                '--evolve-pde-weights', '--no-evolve-pde-weights'}
    if any(item.split('=')[0] in reserved for item in extra):
        parser.error('Study-controlled arguments cannot be overridden in training options')
    common = ['--epochs', str(adam_epochs + args.generations),
              '--de-generations', str(args.generations), '--population-size', str(args.population_size), *extra]
    validated = training_args([*common, '--seed', str(args.seed + args.runs - 1),
                               '--de-seed', str(args.de_seed + args.runs - 1)])
    training_args([*common, '--seed', str(args.seed), '--de-seed', str(args.de_seed)])
    print(f'Per-run budget: {args.budget} = {args.population_size * (args.generations + 1)} DE evaluations + {adam_epochs} Adam updates (maximum).', flush=True)
    require_training_control_api()
    config = args.config_directory.resolve()
    # Check both inputs before creating a study directory.
    for name in ('mesh_properties.json', 'constant_properties.json'):
        json.loads((config / name).read_text())
    root = args.output_directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'config').mkdir()
    (root / 'logs').mkdir()
    for name in ('mesh_properties.json', 'constant_properties.json'):
        shutil.copy2(config / name, root / 'config' / name)
    plan = []
    for index in range(args.runs):
        # Rotate execution order to reduce systematic mode/order confounding.
        order = MODES[index % 4:] + MODES[:index % 4]
        for mode in order:
            label = f'seed_{args.seed + index}_{mode}'
            command = [sys.executable, '-u', '-m', 'runners.train_pinn_de_adam', *common,
                       '--config-directory', str(root / 'config'), '--output-directory', str(root / 'runs' / label),
                       '--seed', str(args.seed + index), '--de-seed', str(args.de_seed + index),
                       '--loss-weight-stage', mode]
            plan.append(dict(label=label, mode=mode, seed=args.seed + index,
                             de_seed=args.de_seed + index, command=command, status='pending',
                             log=f'logs/{label}.log', run=None, error=None))
    manifest = dict(modes=MODES, runs_per_mode=args.runs, generations=args.generations,
                    adam_epochs=adam_epochs, population_size=args.population_size,
                    budget_formula="Adam updates = budget - population_size * (generations + 1)",
                    maximum_evaluations=args.budget,
                    network_seed_start=args.seed, de_seed_start=args.de_seed,
                    training_options={k: str(v) if isinstance(v, Path) else v for k, v in vars(validated).items()},
                    pairing='Same network seed and DE seed across modes; independent seed pairs across replicates',
                    order='Modes rotate by replicate index',
                    threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                    evaluation='Shared float64 FVM with h/2 and k/2; disjoint odd-time validation/test splits',
                    **source_metadata())
    manifest['study_source_sha256'] = {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
        for name in ('runners/run_loss_weight_study.py', 'runners/run_paired_study.py',
                     'utils/analyze_loss_weight_study.py', 'shells/run_loss_weight_study.sh')}
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'records.json', plan)
    print(f'Study: {root}', flush=True)
    try:
        reference = reference_arrays(root)
        if len(reference['t'][3::4]) == 0:
            raise ValueError('Time grid must supply at least one refined test time (index 3)')
        for record in plan:
            record['status'] = 'running'
            write_json(root / 'records.json', plan)
            print(f"START {record['label']}", flush=True)
            start = time.perf_counter()
            try:
                with (root / record['log']).open('w') as stream:
                    result = subprocess.run(record['command'], cwd=BASE, stdout=stream,
                                            stderr=subprocess.STDOUT, check=False)
                record['returncode'] = result.returncode
                runs = list((root / 'runs' / record['label']).glob('*/metadata.json'))
                if len(runs) == 1:
                    record['run'] = str(runs[0].parent.relative_to(root))
                if result.returncode != 0 or len(runs) != 1:
                    raise RuntimeError(f'Training failed (exit={result.returncode}); see {record["log"]}')
                run = root / record['run']
                metadata = json.loads((run / 'metadata.json').read_text())
                if metadata['status'] != 'complete':
                    raise RuntimeError('Trainer did not produce a complete run')
                record['evaluation'] = evaluate(run, reference, test=True)
                record['status'] = 'complete'
            except Exception as error:
                record.update(status='failed', error=f'{type(error).__name__}: {error}')
            finally:
                record['wall_seconds'] = time.perf_counter() - start
                write_json(root / 'records.json', plan)
            print(f"{record['status'].upper()} {record['label']}", flush=True)
    finally:
        # Interrupted/failed and unstarted attempts remain explicit in the report.
        from utils.analyze_loss_weight_study import analyze
        analyze(root)
    return 0 if all(r['status'] == 'complete' for r in plan) else 1


if __name__ == '__main__':
    raise SystemExit(main())
