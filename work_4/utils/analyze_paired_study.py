"""Generate paired study tables and figures directly from stored raw metrics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def table(path, rows):
    if rows:
        with Path(path).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_schedule(record):
    """Read executed DE/Adam budgets and timings from the saved run metadata."""
    if not record.get('run'):
        return {}
    metadata = read(Path(record['run']) / 'metadata.json')
    hybrid = record['method'] == 'pinn-de'
    de = metadata.get('de', {})
    return dict(
        de_population_size=de.get('population_size') if hybrid else None,
        de_generations=de.get('generations') if hybrid else None,
        de_fitness_evaluations=de.get('fitness_evaluations') if hybrid else 0,
        adam_updates=metadata.get('adam_epochs_run', metadata.get('adam_epochs') if hybrid else metadata.get('epochs')),
        actual_evaluations=metadata.get('training_loss_evaluations', record.get('budget')),
        de_seconds=metadata.get('de_seconds') if hybrid else 0.0,
        adam_seconds=metadata.get('adam_seconds') if hybrid else metadata.get('training_seconds'),
    )


def analyze(root, select=False):
    manifest = read(root / 'manifest.json')
    test_title = ('Reused refined-FVM test' if manifest.get('test_reuse_limitation')
                  else 'Reserved refined-FVM test')
    screen = read(root / 'screen_records.json') if (root / 'screen_records.json').exists() else []
    confirm = read(root / 'confirm_records.json') if (root / 'confirm_records.json').exists() else []
    for phase, records in [('screen', screen), ('confirm', confirm)]:
        rows = []
        for record in records:
            values = record.get('evaluation', {})
            row = dict(configuration=record['configuration']['name'], method=record['method'],
                       seed=record['seed'], status='complete' if values else 'failed',
                       architecture=str(record['configuration']['hidden_sizes']), lr=record['configuration']['lr'],
                       evaluations=run_schedule(record).get('actual_evaluations', record['budget']), training_seconds=values.get('training_seconds'),
                       validation_rmse=values.get('validation', {}).get('joint_rmse'),
                       test_rmse=values.get('test', {}).get('joint_rmse'),
                       test_rmse_u=values.get('test', {}).get('rmse_u'),
                       test_rmse_v=values.get('test', {}).get('rmse_v'),
                       initial_rmse=values.get('initial_rmse'), physics_rmse=values.get('physics_joint_rmse'),
                       run=record.get('run'), log=str(root / 'logs' / f'{record["label"]}.log'))
            schedule = run_schedule(record)
            row.update({key: schedule.get(key) for key in (
                'de_population_size', 'de_generations', 'de_fitness_evaluations',
                'adam_updates', 'de_seconds', 'adam_seconds')})
            rows.append(row)
        table(root / f'{phase}_table.csv', rows)
    if screen:
        names = [c['name'] for c in manifest['candidates']]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        scores = {}
        for method, color, offset in [('pinn', 'tab:blue', -.18), ('pinn-de', 'tab:orange', .18)]:
            for panel, key in enumerate(('validation', 'physics_joint_rmse')):
                values = []
                for name in names:
                    matching = [r for r in screen if r['configuration']['name'] == name and r['method'] == method]
                    value = matching[0].get('evaluation', {}).get(key) if matching else None
                    values.append(value['joint_rmse'] if isinstance(value, dict) else (np.nan if value is None else value))
                axes[panel].bar(np.arange(len(names)) + offset, values, width=.36, color=color, label=method)
                axes[panel].set_xticks(np.arange(len(names)), names, rotation=12)
                axes[panel].set(ylabel='RMSE', title='Refined-FVM validation' if panel == 0 else 'Independent PDE residual')
                axes[panel].legend(); axes[panel].grid(axis='y', alpha=.25)
        fig.suptitle('Exploratory screening: one common seed, configured maximum loss-evaluation budgets')
        fig.savefig(root / 'screening.png', dpi=170); plt.close(fig)
        for name in names:
            pair = [r for r in screen if r['configuration']['name'] == name and 'evaluation' in r]
            if len(pair) == 2:
                scores[name] = float(np.mean([r['evaluation']['validation']['joint_rmse'] for r in pair]))
        write(root / 'screening_pair_scores.json', scores)
        if select:
            if len(scores) != len(names):
                raise RuntimeError('Selection requires every planned pair to complete; failed runs remain recorded')
            chosen = min(scores, key=scores.get)
            manifest['selected_configuration'] = next(c for c in manifest['candidates'] if c['name'] == chosen)
            manifest['selection_validation_rmse'] = scores[chosen]
            manifest['status'] = 'confirmation'
            write(root / 'manifest.json', manifest)
            print('Selected:', manifest['selected_configuration'], 'paired validation RMSE:', scores[chosen])
    if confirm:
        summary = []
        for method in ('pinn', 'pinn-de'):
            records = [r for r in confirm if r['method'] == method and 'evaluation' in r]
            row = dict(method=method, runs=len(records), failed_runs=sum(r['method'] == method and 'evaluation' not in r for r in confirm))
            schedules = [run_schedule(record) for record in records]
            for key in ('de_population_size', 'de_generations', 'de_fitness_evaluations', 'adam_updates'):
                unique = {schedule.get(key) for schedule in schedules}
                row[key] = unique.pop() if len(unique) == 1 else None
            for key in ('de_seconds', 'adam_seconds'):
                values = [schedule[key] for schedule in schedules if schedule.get(key) is not None]
                row[key + '_mean'] = float(np.mean(values)) if values else None
                row[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            for key in ('validation', 'test', 'initial_rmse', 'physics_joint_rmse', 'training_seconds'):
                values = [r['evaluation'][key]['joint_rmse'] if isinstance(r['evaluation'][key], dict)
                          else r['evaluation'][key] for r in records]
                row[key + '_mean'] = float(np.mean(values)) if values else None
                row[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.append(row)
        table(root / 'confirmation_summary.csv', summary)
        write(root / 'confirmation_summary.json', summary)
        complete = [r for r in confirm if 'evaluation' in r]
        if complete:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
            for method, color in [('pinn', 'tab:blue'), ('pinn-de', 'tab:orange')]:
                records = [r for r in complete if r['method'] == method]
                if not records:
                    continue
                for ax, key in zip(axes, ('test', 'physics_joint_rmse')):
                    values = [r['evaluation'][key]['joint_rmse'] if key == 'test' else r['evaluation'][key] for r in records]
                    ax.plot([r['seed'] for r in records], values, 'o-', label=method, color=color)
                    ax.set(xlabel='Independent network seed', ylabel='RMSE', title=test_title if key == 'test' else 'Independent PDE residual')
                    ax.ticklabel_format(style='plain', axis='x', useOffset=False)
                    ax.set_xticks(sorted({r['seed'] for r in complete}))
                    ax.grid(alpha=.25);ax.legend()
            fig.savefig(root / 'confirmation.png', dpi=170);plt.close(fig)
            fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
            for method, color in [('pinn', 'tab:blue'), ('pinn-de', 'tab:orange')]:
                records = [r for r in complete if r['method'] == method]
                if not records:
                    continue
                values = np.asarray([r['evaluation']['test']['rmse_by_time'] for r in records])
                t = records[0]['evaluation']['test']['times']; mean = values.mean(0)
                ax.plot(t, mean, color=color, label=f'{method} (n={len(records)})')
                if len(records) > 1:
                    std = values.std(0, ddof=1);ax.fill_between(t, np.maximum(0, mean-std), mean+std, color=color, alpha=.2)
            ax.set(xlabel='Physical time', ylabel='Joint u/v RMSE', title=test_title + ': mean and sample standard deviation')
            ax.grid(alpha=.25);ax.legend();fig.savefig(root / 'test_error_by_time.png', dpi=170);plt.close(fig)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--select', action='store_true')
    args = parser.parse_args()
    analyze(args.study.resolve(), args.select)
