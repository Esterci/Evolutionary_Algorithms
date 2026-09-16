"""Compare recorded small- and large-budget PINN/DE confirmation runs.

All figures and tables are derived from saved metrics. This command does not
train models, select checkpoints, or modify the previous study's artifacts.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.analyze_paired_study import analyze, read, run_schedule, table, write


def compare(root):
    manifest = read(root / 'manifest.json')
    previous = Path(manifest['previous_study'])
    analyze(root)
    rows, diagnostics, histories = [], [], []
    for label, study in [('previous', previous), ('followup', root)]:
        records_path = study / 'confirm_records.json'
        records = read(records_path) if records_path.exists() else []
        for record in records:
            evaluation = record.get('evaluation', {})
            schedule = run_schedule(record)
            metadata = read(Path(record['run']) / 'metadata.json') if record.get('run') else {}
            row = dict(
                study=label, method=record['method'], seed=record['seed'],
                status='complete' if evaluation else 'failed',
                hidden_sizes=str(record['configuration']['hidden_sizes']),
                learning_rate=record['configuration']['lr'],
                total_evaluations=schedule.get('actual_evaluations', record['budget']),
                de_population_size=schedule.get('de_population_size'),
                de_generations=schedule.get('de_generations'),
                de_evaluations=schedule.get('de_fitness_evaluations'),
                adam_updates=schedule.get('adam_updates'),
                validation_rmse=evaluation.get('validation', {}).get('joint_rmse'),
                test_rmse=evaluation.get('test', {}).get('joint_rmse'),
                test_rmse_u=evaluation.get('test', {}).get('rmse_u'),
                test_rmse_v=evaluation.get('test', {}).get('rmse_v'),
                pde_rmse=evaluation.get('physics_joint_rmse'),
                initial_rmse=evaluation.get('initial_rmse'),
                de_seconds=schedule.get('de_seconds'),
                adam_seconds=schedule.get('adam_seconds'),
                training_seconds=evaluation.get('training_seconds'),
                run=record.get('run'),
            )
            for name in ('Initial', 'Boundary', 'PDE'):
                row['final_weight_' + name] = metadata.get('final_loss_weights', {}).get(name)
            rows.append(row)
            if not evaluation or record['method'] != 'pinn-de':
                continue
            run = Path(record['run'])
            with (run / 'de_history.csv').open() as stream:
                history = list(csv.DictReader(stream))
            first, last = history[0], history[-1]
            diagnostic = dict(study=label, seed=record['seed'],
                              population_size=schedule['de_population_size'],
                              generations=schedule['de_generations'],
                              de_evaluations=schedule['de_fitness_evaluations'])
            for key in ('best_fitness', 'selected_reference_fitness',
                        'best_reference_fitness', 'population_spread'):
                diagnostic[key + '_initial'] = float(first[key])
                diagnostic[key + '_final'] = float(last[key])
            for key in ('accepted_trials', 'control_adaptation_attempts',
                        'control_adaptation_acceptances'):
                diagnostic[key + '_total'] = sum(int(h[key]) for h in history)
            for name in ('Initial', 'Boundary', 'PDE'):
                diagnostic['de_final_loss_' + name] = float(last['best_loss_' + name])
                diagnostic['de_final_weight_' + name] = float(last['best_weight_' + name])
                diagnostic['adam_final_loss_' + name] = metadata['final_losses'][name]
                diagnostic['adam_final_weight_' + name] = metadata['final_loss_weights'][name]
            diagnostics.append(diagnostic)
            histories.append((label, record['seed'], history))
    table(root / 'budget_comparison_runs.csv', rows)
    table(root / 'de_diagnostics_summary.csv', diagnostics)
    write(root / 'de_diagnostics_summary.json', diagnostics)

    summaries = []
    metrics = ('validation_rmse', 'test_rmse', 'test_rmse_u', 'test_rmse_v',
               'pde_rmse', 'initial_rmse', 'de_seconds', 'adam_seconds', 'training_seconds')
    for study in ('previous', 'followup'):
        for method in ('pinn', 'pinn-de'):
            group = [r for r in rows if r['study'] == study and r['method'] == method]
            complete = [r for r in group if r['status'] == 'complete']
            summary = dict(study=study, method=method, runs=len(complete),
                           failed_runs=len(group)-len(complete))
            for key in ('total_evaluations', 'de_population_size', 'de_generations',
                        'de_evaluations', 'adam_updates'):
                values = {r[key] for r in complete}
                summary[key] = values.pop() if len(values) == 1 else None
            for metric in metrics:
                values = [r[metric] for r in complete if r[metric] is not None]
                summary[metric + '_mean'] = float(np.mean(values)) if values else None
                summary[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summaries.append(summary)
    table(root / 'budget_comparison_summary.csv', summaries)
    write(root / 'budget_comparison_summary.json', summaries)

    def cell(value):
        return 'N/A' if value is None else f'{value:,}'

    def statistic(row, metric):
        mean, std = row[metric + '_mean'], row[metric + '_std']
        if mean is None:
            return 'Pending'
        digits = 1 if metric.endswith('seconds') else 6
        return f'{mean:.{digits}f}' if std is None else f'{mean:.{digits}f} ± {std:.{digits}f}'

    lines = [
        '| Study | Method | Completed runs | DE individuals | DE generations | DE evaluations | Adam updates | Total evaluations |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summaries:
        values = [row['study'], row['method'], str(row['runs'])]
        values += [cell(row[key]) for key in ('de_population_size', 'de_generations',
                                              'de_evaluations', 'adam_updates', 'total_evaluations')]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', 'Metrics are means ± sample standard deviations; run counts are shown above.', '',
              '| Study | Method | Reused test RMSE | PDE RMSE | Initial RMSE | Training seconds |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in summaries:
        values = [row['study'], row['method']]
        values += [statistic(row, key) for key in ('test_rmse', 'pde_rmse', 'initial_rmse', 'training_seconds')]
        lines.append('| ' + ' | '.join(values) + ' |')
    (root / 'budget_tables.md').write_text('\n'.join(lines) + '\n')

    changes = []
    for current in [r for r in rows if r['study'] == 'followup' and r['status'] == 'complete']:
        matches = [r for r in rows if r['study'] == 'previous' and r['status'] == 'complete'
                   and r['seed'] == current['seed'] and r['method'] == current['method']]
        if not matches:
            continue
        previous_row = matches[0]
        change = dict(method=current['method'], seed=current['seed'])
        for metric in ('test_rmse', 'pde_rmse', 'initial_rmse', 'training_seconds'):
            delta = current[metric] - previous_row[metric]
            change[metric + '_previous'] = previous_row[metric]
            change[metric + '_followup'] = current[metric]
            change[metric + '_delta'] = delta
            change[metric + '_percent_change'] = 100 * delta / previous_row[metric]
        changes.append(change)
    table(root / 'budget_changes_by_seed.csv', changes)
    write(root / 'budget_changes_by_seed.json', changes)

    prefix_checks = []
    for current in [r for r in rows if r['study'] == 'followup' and r['method'] == 'pinn'
                    and r['status'] == 'complete']:
        matches = [r for r in rows if r['study'] == 'previous' and r['method'] == 'pinn'
                   and r['seed'] == current['seed'] and r['status'] == 'complete']
        if not matches:
            continue
        with np.load(Path(matches[0]['run']) / 'losses.npz') as old_losses, \
                np.load(Path(current['run']) / 'losses.npz') as new_losses:
            terms = {}
            for name in ('Initial', 'Boundary', 'PDE'):
                shared = min(len(old_losses[name]), len(new_losses[name]))
                a, b = old_losses[name][:shared], new_losses[name][:shared]
                terms[name] = dict(shared_updates=shared, exactly_equal=bool(np.array_equal(a, b)),
                                   maximum_absolute_difference=float(np.max(np.abs(a-b))))
        prefix_checks.append(dict(seed=current['seed'],loss_history=terms))
    write(root / 'pinn_prefix_checks.json', prefix_checks)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for study, offset, color in [('previous', -.18, 'tab:blue'), ('followup', .18, 'tab:orange')]:
        group = [r for r in summaries if r['study'] == study]
        for ax, metric, title in zip(axes, ('test_rmse', 'pde_rmse', 'training_seconds'),
                                     ('Reused refined-FVM test', 'Independent PDE residual', 'Measured training time')):
            values = [r[metric + '_mean'] if r[metric + '_mean'] is not None else np.nan for r in group]
            errors = [r[metric + '_std'] or 0.0 for r in group]
            label = ('5,000 evaluations; DE pop. 8, gen. 20' if study == 'previous'
                     else '8,064 evaluations; DE pop. 32, gen. 100')
            ax.bar(np.arange(2)+offset, values, width=.36, yerr=errors, capsize=4, color=color, label=label)
            ax.set_xticks([0, 1], ['PINN', 'PINN-DE'])
            ax.set(title=title, ylabel='Seconds' if metric == 'training_seconds' else 'RMSE')
            ax.grid(axis='y', alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Previous vs follow-up: mean and sample standard deviation; see tables for completed run counts')
    fig.savefig(root / 'budget_comparison.png', dpi=170)
    plt.close(fig)

    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        seeds = sorted({seed for _, seed, _ in histories})
        for study, seed, history in histories:
            evaluations = [int(h['fitness_evaluations']) for h in history]
            style = '--' if study == 'previous' else '-'
            label = f'{seed}: ' + ('8 individuals / 20 generations' if study == 'previous' else '32 individuals / 100 generations')
            for ax, key, title in zip(axes, ('best_fitness', 'selected_reference_fitness'),
                                     ('Best adaptive DE objective', 'Selected genome: fixed-reference loss')):
                ax.plot(evaluations, [float(h[key]) for h in history], style,
                        color=f'C{seeds.index(seed)}', label=label)
                ax.set(xlabel='DE fitness evaluations', ylabel='Objective', title=title)
                ax.grid(alpha=.25)
        axes[0].legend(fontsize=7)
        fig.savefig(root / 'de_budget_diagnostics.png', dpi=170)
        plt.close(fig)
    print(f'Budget comparison artifacts: {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    args = parser.parse_args()
    compare(args.study.resolve())
