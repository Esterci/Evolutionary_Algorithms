"""Run a recorded, paired PINN/DE study using the existing training entry points.

The physical training configuration is never changed. Refined FVM values are
used only for validation/test evaluation, not for any training loss.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from pinn_model import BurgersProblem, NormalizedFisiocomPINN, predict_in_batches
from train_pinn import compare_fvm
from train_pinn_de_adam import physics_validation

BASE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def reference_arrays(root):
    path = root / 'refined_fvm.npz'
    if path.exists():
        with np.load(path) as arrays:
            return dict(arrays)
    mesh = read_json(root / 'config/mesh_properties.json')
    constants = read_json(root / 'config/constant_properties.json')
    mesh.update(h=mesh['h'] / 2, k=mesh['k'] / 2)
    problem = BurgersProblem(mesh, constants, torch.float64)
    # compare_fvm's reference is independent of the model predictions.
    model = NormalizedFisiocomPINN(problem, [1]).to(dtype=torch.float64)
    arrays, _, fvm_seconds, _ = compare_fvm(problem, model, 4096)
    arrays = {name: arrays[name] for name in ('x', 'y', 't', 'fvm_u', 'fvm_v')}
    np.savez_compressed(path, **arrays)
    write_json(root / 'refined_fvm_metadata.json', dict(
        mesh=mesh, constants=constants, dtype='float64', fvm_seconds=fvm_seconds,
        purpose='independent validation/test reference; never enters training losses',
        limitation='first-order FVM point/cell comparison; refinement is not an accuracy proof'))
    return arrays


def evaluate(run, reference, test=False):
    metadata = read_json(run / 'metadata.json')
    problem = BurgersProblem(metadata['mesh'], metadata['constants'], torch.float32)
    model = NormalizedFisiocomPINN(problem, metadata['architecture']).to('cuda')
    model.load_state_dict(torch.load(run / 'pinn_model.pt', map_location='cuda', weights_only=True))
    model.eval()
    xx, yy = np.meshgrid(reference['x'], reference['y'], indexing='ij')
    splits = dict(validation=np.arange(1, len(reference['t']), 4))
    if test:
        splits['test'] = np.arange(3, len(reference['t']), 4)
    results = {}
    for name, indices in splits.items():
        differences, targets = [], []
        for index in indices:
            inputs = np.column_stack((np.full(xx.size, reference['t'][index]), xx.ravel(), yy.ravel()))
            prediction = predict_in_batches(model, inputs, 4096).reshape(*xx.shape, 2)
            target = np.stack((reference['fvm_u'][index], reference['fvm_v'][index]), axis=-1)
            differences.append(prediction - target)
            targets.append(target)
        difference, target = np.asarray(differences), np.asarray(targets)
        results[name] = dict(
            joint_rmse=float(np.sqrt(np.mean(difference**2))),
            rmse_u=float(np.sqrt(np.mean(difference[..., 0]**2))),
            rmse_v=float(np.sqrt(np.mean(difference[..., 1]**2))),
            relative_l2_u=float(np.linalg.norm(difference[..., 0]) / np.linalg.norm(target[..., 0])),
            relative_l2_v=float(np.linalg.norm(difference[..., 1]) / np.linalg.norm(target[..., 1])),
            maximum_absolute=float(np.max(np.abs(difference))),
            time_indices=indices.tolist(), times=reference['t'][indices].tolist(),
            rmse_by_time=np.sqrt(np.mean(difference**2, axis=(1, 2, 3))).tolist(),
            points=int(difference.size // 2))
    inputs = np.column_stack((np.full(xx.size, reference['t'][0]), xx.ravel(), yy.ravel()))
    predicted_initial = predict_in_batches(model, inputs, 4096).reshape(*xx.shape, 2)
    initial_target = np.stack((reference['fvm_u'][0], reference['fvm_v'][0]), axis=-1)
    results['initial_rmse'] = float(np.sqrt(np.mean((predicted_initial - initial_target)**2)))
    results['physics'] = physics_validation(problem, model)
    results['physics_joint_rmse'] = math.sqrt(np.mean(np.square(results['physics']['rmse'])))
    results['training_seconds'] = metadata.get('training_seconds', 0) + metadata.get('de_seconds', 0) + metadata.get('adam_seconds', 0)
    write_json(run / 'study_evaluation.json', results)
    del model
    torch.cuda.empty_cache()
    return results


def run_study(root, phase):
    manifest = read_json(root / 'manifest.json')
    reference = reference_arrays(root)
    if phase == 'screen':
        configurations = manifest['candidates']
        seeds, budget = [manifest['network_seed_screening']], manifest['screening_budget']
    else:
        configurations = [manifest['selected_configuration']]
        seeds, budget = manifest['confirmation_seeds'], manifest['confirmation_budget']
    records_path = root / f'{phase}_records.json'
    records = read_json(records_path) if records_path.exists() else []
    for configuration in configurations:
        for seed in seeds:
            for method in ('pinn', 'pinn-de'):
                label = f'{phase}_{configuration["name"]}_s{seed}_{method}'
                if any(record['label'] == label for record in records):
                    continue
                parent = root / 'runs' / label
                script = 'train_pinn.py' if method == 'pinn' else 'train_pinn_de_adam.py'
                de = dict(manifest['de'], **configuration.get('de', {}))
                de_budget = de['population_size'] * (de['generations'] + 1)
                epochs = budget if method == 'pinn' else budget - de_budget + de['generations']
                de_seed = manifest['de_seed'] if phase == 'screen' else seed + 1000
                command = [sys.executable, '-u', str(BASE / script), '--config-directory', str(root / 'config'),
                           '--output-directory', str(parent), '--epochs', str(epochs),
                           '--hidden-sizes', *map(str, configuration['hidden_sizes']), '--lr', str(configuration['lr']),
                           '--seed', str(seed), '--device', 'cuda', '--dtype', 'float32']
                if method == 'pinn-de':
                    command += ['--de-generations', str(de['generations']), '--population-size', str(de['population_size']),
                                '--initial-spread', str(de['initial_spread']), '--de-seed', str(de_seed),
                                '--control-adaptation', de['control_adaptation'], '--adaptation-probability', str(de['adaptation_probability'])]
                print(f'START {label}: {budget} training loss evaluations', flush=True)
                start = time.perf_counter()
                with (root / 'logs' / f'{label}.log').open('w') as stream:
                    completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
                record = dict(label=label, phase=phase, configuration=configuration, seed=seed, method=method,
                              budget=budget, de_seed=de_seed if method == 'pinn-de' else None,
                              command=command, returncode=completed.returncode,
                              wall_seconds=time.perf_counter() - start)
                runs = sorted(parent.glob('*/metadata.json'))
                if completed.returncode == 0 and len(runs) == 1:
                    run = runs[0].parent
                    record['run'] = str(run)
                    record['evaluation'] = evaluate(run, reference, test=phase == 'confirm')
                    print(f'END {label}: validation RMSE {record["evaluation"]["validation"]["joint_rmse"]:.6g}; '
                          f'physics RMSE {record["evaluation"]["physics_joint_rmse"]:.6g}; wall {record["wall_seconds"]:.1f}s', flush=True)
                else:
                    record['error'] = 'Training failed; see preserved log and partial artifacts'
                    print(f'FAILED {label}; preserving run', flush=True)
                records.append(record)
                write_json(records_path, records)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--phase', choices=['screen', 'confirm'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('This recorded study requires the GPU configured in its manifest')
    run_study(args.study.resolve(), args.phase)
