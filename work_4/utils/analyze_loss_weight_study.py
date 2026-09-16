"""Summarize stored four-mode experiments without retraining or importing torch."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.analyze_paired_study import table, write

MODES = ('de', 'pinn', 'both', 'none')
METRICS = ('validation_rmse', 'test_rmse', 'physics_rmse', 'initial_rmse',
           'boundary_training_mse', 'training_seconds', 'adam_updates', 'evaluations')
COLORS = dict(zip(MODES, ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')))


def statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0, mean=None, std=None, median=None, min=None, max=None, iqr=None)
    if not np.isfinite(values).all():
        raise ValueError('Cannot aggregate non-finite values')
    return dict(n=len(values), mean=float(values.mean()),
                std=float(values.std(ddof=1)) if len(values) > 1 else None,
                median=float(np.median(values)), min=float(values.min()), max=float(values.max()),
                iqr=float(np.percentile(values, 75) - np.percentile(values, 25)))


def mean_history(curves):
    """Average a fixed cohort over its common prefix, without extending stopped runs."""
    arrays = [np.asarray(curve, dtype=float) for curve in curves]
    if not arrays or any(a.ndim != 1 or not len(a) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Expected nonempty finite one-dimensional histories')
    length = min(map(len, arrays))
    values = np.stack([a[:length] for a in arrays])
    return values.mean(0), values.std(0, ddof=1) if len(values) > 1 else None


def weighted_loss_history(loss, post_update_weights, initial_weight):
    """Multiply pre-update MSE by the coefficient used at that same update."""
    loss = np.asarray(loss, dtype=float)
    weights = np.asarray(post_update_weights, dtype=float)
    if loss.ndim != 1 or not len(loss) or weights.shape != loss.shape:
        raise ValueError('Loss and weight histories must be nonempty aligned vectors')
    if (not np.isfinite(loss).all() or not np.isfinite(weights).all()
            or not np.isfinite(initial_weight) or initial_weight <= 0
            or np.any(weights <= 0)):
        raise ValueError('Expected finite losses and positive finite coefficients')
    before_update = np.concatenate(([initial_weight], weights[:-1]))
    return loss * before_update


def select_best_runs(completed, per_mode=True):
    """Select by validation RMSE; break ties by seed, never by test error."""
    groups = MODES if per_mode else (None,)
    selected = []
    for mode in groups:
        candidates = [(record, row) for record, row in completed
                      if mode is None or record['mode'] == mode]
        if candidates:
            selected.append(min(candidates, key=lambda pair: (
                pair[1]['validation_rmse'], pair[0]['seed'], pair[0]['mode'])))
    return selected


def plot_mean_histories(root, completed, output):
    """Plot DE seed means and selected PINN trajectories; export plotted values."""
    exports = []
    datasets = {}
    terms = ('Initial', 'Boundary', 'PDE')
    reference_coefficients = dict.fromkeys(terms, 1.0)
    for mode in MODES:
        datasets[mode] = []
        for record, _ in completed:
            if record['mode'] != mode:
                continue
            directory = root / record['run']
            metadata = json.loads((directory / 'metadata.json').read_text())
            with np.load(directory / 'losses.npz') as data:
                losses = {name: data[name].copy() for name in terms}
            with np.load(directory / 'de_history.npz') as data:
                de = {name: data[name].copy() for name in data.files}
            weights_path = directory / 'adam_loss_weights.npz'
            if weights_path.exists():
                with np.load(weights_path) as data:
                    names = list(data['loss_names'])
                    weights = {name: data['weights'][:, names.index(name)].copy() for name in terms}
            else:
                if mode != 'none':
                    raise ValueError(f'Missing Adam weight history: {directory}')
                weights = {name: np.full(len(losses[name]), metadata['loss_weights'][name]) for name in terms}
            if mode != 'none' and metadata['adam_weight_history_timing'] != 'after each Adam update':
                raise ValueError('Unsupported weight-history timing')
            initial_weights = (
                {name: np.exp(-metadata['adam_initial_loss_log_vars'][name]) for name in terms}
                if mode != 'none' else metadata['loss_weights']
            )
            weighted = {name: weighted_loss_history(losses[name], weights[name], initial_weights[name])
                        for name in terms}
            # Shared post-hoc reference, including for the fixed-coefficient mode.
            coefficients = reference_coefficients
            reference = {name: coefficients[name] * losses[name] for name in terms}
            if len({len(values) for values in reference.values()}) != 1:
                raise ValueError('Loss histories must share Adam update indices')
            reference['Total'] = sum(reference.values())
            datasets[mode].append(dict(record=record, losses=losses, de=de, weights=weights,
                                       reference=reference, weighted=weighted))

    best = {record['mode']: record for record, _ in select_best_runs(completed)}
    selections = [dict(mode=record['mode'], seed=record['seed'], de_seed=record['de_seed'],
                       run=record['run'], validation_rmse=row['validation_rmse'],
                       criterion='minimum validation RMSE; ties resolved by seed')
                  for record, row in select_best_runs(completed)]
    write(output / 'selected_pinn_runs.json', selections)

    def draw(axis, mode, source, metric, label, color=None, style='-', de_step='fitness_evaluations'):

        runs = datasets[mode]
        selected = source != 'de'
        if selected:
            runs = [run for run in runs if run['record'] == best.get(mode)]
        if not runs:
            return
        mean, sd = mean_history([r[source][metric] for r in runs])
        x = runs[0]['de'][de_step][:len(mean)] if source == 'de' else np.arange(1, len(mean) + 1)
        if source == 'de' and any(not np.array_equal(x, r['de'][de_step][:len(mean)]) for r in runs):
            raise ValueError('Incompatible DE evaluation grids')
        legend = (f"{label} | seed={runs[0]['record']['seed']} | DE seed={runs[0]['record']['de_seed']}"
                  if selected else f'{label} (n={len(runs)})')
        line, = axis.plot(x, mean, style, color=color, label=legend)
        if sd is not None:
            axis.fill_between(x, mean - sd, mean + sd, color=line.get_color(), alpha=.13)
        seeds = ','.join(str(r['record']['seed']) for r in runs)
        for i, value in enumerate(mean):
            exports.append(dict(mode=mode, source=source, metric=metric, step=int(x[i]),
                                step_kind=de_step if source == 'de' else 'adam_update',
                                aggregation='best_validation_run' if selected else 'mean_across_seeds',
                                n=len(runs), seeds=seeds, mean=float(value),
                                std=None if sd is None else float(sd[i])))

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    for row, term in enumerate(terms):
        for column, source in enumerate(('reference', 'weighted')):
            axis = axes[row, column]
            for mode in MODES:
                draw(axis, mode, source, term, 'none (fixed coefficients)' if mode == 'none' and source == 'weighted' else mode, COLORS[mode])
            label = 'Fixed reference (1 × MSE)' if source == 'reference' else 'Adapted coefficient × MSE'
            axis.set(title=f'{term}: {label}', xlabel='Adam update (loss before update)',
                     ylabel=label)
            axis.set_yscale('symlog', linthresh=1e-8)
            axis.grid(alpha=.2)
            if completed:
                axis.legend(fontsize=8)
    fig.suptitle('Fixed vs adapted losses: best validation run per mode\n'
                 'Weighted MSE contributions; log-variance regularization excluded')
    fig.savefig(output / 'training_losses.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, term in zip(axes, terms):
        for mode in ('de', 'pinn', 'both'):
            draw(axis, mode, 'weights', term, mode, COLORS[mode])
        axis.set(title=term, xlabel='Adam update', ylabel='Loss coefficient after Adam update')
        axis.set_yscale('symlog', linthresh=1e-8)
        axis.grid(alpha=.2)
        if completed:
            axis.legend(fontsize=8)
    fig.suptitle('PINN coefficients: best validation run per mode')
    fig.savefig(output / 'pinn_loss_weights.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, mode in zip(axes.flat, MODES):
        for metric, label in [('best_fitness', 'Best fitness'), ('mean_fitness', 'Population mean fitness')]:
            draw(axis, mode, 'de', metric, label)
        axis.set(title=mode, xlabel='DE fitness evaluations (including initialization)', ylabel='DE objective')
        axis.set_yscale('symlog', linthresh=1)
        axis.grid(alpha=.2)
        if datasets[mode]:
            axis.legend(fontsize=8)
    fig.suptitle('DE convergence: mean ± sample SD across seeds; objectives differ across modes')
    fig.savefig(output / 'de_convergence.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for row, mode in enumerate(('de',)):
        for metric, label in [('selected_reference_fitness', 'DE-selected candidate'),
                              ('best_reference_fitness', 'Best reference among evaluated candidates')]:
            draw(axes[row, 0], mode, 'de', metric, label)
        for term in terms:
            draw(axes[row, 1], mode, 'de', 'best_weight_' + term, term)
        for col, ylabel in enumerate(('Fixed reference: MSE(IC) + MSE(BC) + MSE(PDE)', 'DE-selected loss coefficient')):
            axis = axes[row, col]
            axis.set(title=mode, xlabel='DE fitness evaluations', ylabel=ylabel)
            axis.set_yscale('symlog', linthresh=1e-8)
            axis.grid(alpha=.2)
            if datasets[mode]:
                axis.legend(fontsize=8)
    for axis, metric, label in zip(axes[1], ('mean_mutation_factor', 'mean_crossover_rate'),
                                   ('Mutation factor F', 'Crossover rate CR')):
        for mode in ('de', 'both'):
            draw(axis, mode, 'de', metric, mode, COLORS[mode],
                 style='--' if mode == 'both' else '-', de_step='generation')
        axis.set(title=label, xlabel='DE generation (0 = initialization)',
                 ylabel='Population mean, averaged across seeds')
        axis.grid(alpha=.2)
        if any(datasets[mode] for mode in ('de', 'both')):
            axis.legend(fontsize=8)
    fig.suptitle('DE diagnostics: mean ± sample SD across seeds')
    fig.savefig(output / 'de_diagnostics.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, term in zip(axes.flat, (*terms, 'Total')):
        for mode in MODES:
            draw(axis, mode, 'reference', term, mode, COLORS[mode])
        coefficient = (reference_coefficients or {}).get(term)
        title = 'Sum of fixed-reference contributions' if term == 'Total' else f'{term}: {coefficient} × raw MSE'
        axis.set(title=title, xlabel='Adam update (loss before update)', ylabel='Fixed-reference training objective')
        axis.set_yscale('symlog', linthresh=1e-8)
        axis.grid(alpha=.2)
        if completed:
            axis.legend(fontsize=8)
    fig.suptitle('PINN fixed reference: best validation run per mode')
    fig.savefig(output / 'pinn_fixed_reference.png', dpi=160)
    plt.close(fig)
    if exports:
        unique = {(r['mode'], r['source'], r['metric'], r['step']): r for r in exports}
        table(output / 'mean_histories.csv', list(unique.values()))


def analyze(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    records = json.loads((root / 'records.json').read_text())
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    rows, completed = [], []
    seen = set()
    for record in records:
        key = (record['mode'], record['seed'])
        if key in seen or record['mode'] not in MODES:
            raise ValueError(f'Duplicate or invalid run: {key}')
        seen.add(key)
        row = dict(mode=record['mode'], seed=record['seed'], de_seed=record['de_seed'],
                   status=record['status'], error=record.get('error'),
                   run=record.get('run'), log=record['log'], stopped_early=None,
                   **dict.fromkeys(METRICS))
        if record['status'] == 'complete':
            try:
                metadata = json.loads((root / record['run'] / 'metadata.json').read_text())
                value = record['evaluation']
                row.update(validation_rmse=value['validation']['joint_rmse'],
                           test_rmse=value['test']['joint_rmse'], physics_rmse=value['physics_joint_rmse'],
                           initial_rmse=value['initial_rmse'], boundary_training_mse=metadata['final_losses']['Boundary'],
                           training_seconds=value['training_seconds'],
                           adam_updates=metadata['adam_epochs_run'], evaluations=metadata['training_loss_evaluations'],
                           stopped_early=metadata['early_stopping']['stopped_early'])
                if not all(np.isfinite(row[m]) for m in METRICS):
                    raise ValueError('Non-finite result metric')
                if metadata['loss_weight_stage'] != record['mode'] or metadata['seed'] != record['seed']:
                    raise ValueError('Run metadata disagrees with planned mode/seed')
                completed.append((record, row))
            except (KeyError, ValueError, OSError, TypeError) as error:
                row.update(status='invalid', error=str(error), **dict.fromkeys(METRICS))
        rows.append(row)
    summaries = []
    for mode in MODES:
        selected = [row for row in rows if row['mode'] == mode and row['status'] == 'complete']
        for metric in METRICS:
            summaries.append(dict(mode=mode, metric=metric, planned=manifest['runs_per_mode'],
                                  incomplete=manifest['runs_per_mode'] - len(selected),
                                  **statistics([r[metric] for r in selected])))
    pairs = []
    for left, right in itertools.combinations(MODES, 2):
        left_runs = {r['seed']: r for r in rows if r['mode'] == left and r['status'] == 'complete'}
        right_runs = {r['seed']: r for r in rows if r['mode'] == right and r['status'] == 'complete'}
        for seed in sorted(left_runs.keys() & right_runs.keys()):
            pairs.append(dict(left=left, right=right, seed=seed,
                              **{m: left_runs[seed][m] - right_runs[seed][m] for m in METRICS}))
    table(output / 'runs.csv', rows)
    table(output / 'summary.csv', summaries)
    # Always replace derived paired tables, including the zero-pair case.
    if pairs:
        table(output / 'paired_differences.csv', pairs)
    else:
        (output / 'paired_differences.csv').write_text('left,right,seed,' + ','.join(METRICS) + '\n')
    write(output / 'summary.json', summaries)
    write(output / 'runs.json', rows)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, ('test_rmse', 'validation_rmse', 'physics_rmse',
                                       'initial_rmse', 'boundary_training_mse', 'training_seconds')):
        for index, mode in enumerate(MODES):
            data = [r[metric] for r in rows if r['mode'] == mode and r['status'] == 'complete']
            axis.scatter(np.full(len(data), index), data, color=COLORS[mode], alpha=.65)
            if data:
                axis.errorbar(index, np.mean(data), yerr=np.std(data, ddof=1) if len(data) > 1 else None,
                              fmt='D', color='black', capsize=5)
        # Connect paired seeds, with gaps for failed/missing modes.
        for seed in sorted({r['seed'] for r in rows}):
            lookup = {r['mode']: r[metric] for r in rows if r['seed'] == seed and r['status'] == 'complete'}
            axis.plot(range(4), [lookup.get(m, np.nan) for m in MODES], color='gray', alpha=.25)
        axis.set_xticks(range(4), MODES)
        axis.set_ylabel({'training_seconds': 'Training time (s)',
                         'boundary_training_mse': 'Final boundary MSE (training grid)'}.get(metric, metric.replace('_', ' ')))
        axis.grid(alpha=.2)
    fig.suptitle('Loss-weight adaptation: individual runs, mean and sample SD')
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for mode in MODES:
        evaluations = [r['evaluation']['test'] for r, _ in completed if r['mode'] == mode]
        if not evaluations:
            continue
        times = np.asarray(evaluations[0]['times'])
        if any(not np.array_equal(times, e['times']) for e in evaluations):
            raise ValueError('Cannot aggregate incompatible test time grids')
        curves = np.array([e['rmse_by_time'] for e in evaluations])
        axis.plot(times, curves.mean(0), color=COLORS[mode], label=f'{mode} (n={len(curves)})')
        if len(curves) > 1:
            spread = curves.std(0, ddof=1)
            axis.fill_between(times, curves.mean(0) - spread, curves.mean(0) + spread,
                              color=COLORS[mode], alpha=.15)
    axis.set(xlabel='Physical time (model units)', ylabel='Joint FVM-reference RMSE',
             title='Refined-grid test error: mean ± sample SD')
    if completed:
        axis.legend()
    axis.grid(alpha=.2)
    fig.savefig(output / 'test_error_by_time.png', dpi=160)
    plt.close(fig)
    plot_mean_histories(root, completed, output)
    # Choose the first planned seed, never the best observed run.
    representative = [r for r, _ in completed if r['seed'] == manifest['network_seed_start']]
    field_images = []
    if representative:
        from utils.plot_pinn_results import load_comparison, comparison_panel, validate_runs
        runs = [dict(label=f"{r['mode']} | network seed={r['seed']}\nDE seed={r['de_seed']}", directory=root / r['run'], arrays=load_comparison(root / r['run'] / 'comparison.npz'),
                     metadata=json.loads((root / r['run'] / 'metadata.json').read_text()))
                for r in representative]
        validate_runs(runs)
        arrays = runs[0]['arrays']
        for variable in ('u', 'v'):
            figure, update = comparison_panel(arrays, {r['label']: r['arrays'] for r in runs},
                                              variable=variable)
            for index in np.unique(np.linspace(0, len(arrays['t']) - 1, 3, dtype=int)):
                update(index)
                suffix = '' if variable == 'u' else '_v'
                filename = f'fields_seed_{manifest["network_seed_start"]}_time_{index:05d}{suffix}.png'
                figure.savefig(output / filename, dpi=140)
                field_images.append(filename)
            plt.close(figure)
    lines = ['# Loss-weight adaptation study', '',
             f"Planned: {manifest['runs_per_mode']} independent runs per mode; "
             f"{manifest['generations']} DE generations, at most {manifest['adam_epochs']} Adam updates.", '',
             f"Maximum total loss evaluations per run: {manifest.get('maximum_evaluations', 'not recorded')}. Actual costs are listed below.", '',
             '| Mode | Completed / planned | Test RMSE (mean ± sample SD) | Training time (s) |',
             '| --- | --- | --- | --- |']
    for mode in MODES:
        test = next(s for s in summaries if s['mode'] == mode and s['metric'] == 'test_rmse')
        timing = next(s for s in summaries if s['mode'] == mode and s['metric'] == 'training_seconds')
        score = 'not computed' if test['n'] == 0 else f"{test['mean']:.6g} ± " + (f"{test['std']:.6g}" if test['std'] is not None else 'N/A (n=1)')
        duration = 'not computed' if timing['mean'] is None else f"{timing['mean']:.4g}"
        lines.append(f"| {mode} | {test['n']} / {test['planned']} | {score} | {duration} |")
    lines += ['', '![Comparison](comparison.png)', '', '![Test error](test_error_by_time.png)', '',
              'Comparison includes final raw boundary MSE on the training boundary grid; '
              'this is not an independent boundary-validation metric. Evaluations remain in the tables.', '',
              '![Training losses](training_losses.png)', '',
              'Each training-loss row pairs a common fixed 1:1:1 reference (left) with the actual '
              'coefficient times MSE (right). de uses DE-adapted coefficients frozen during Adam; '
              'pinn and both adapt them during Adam; none retains its configured fixed coefficients. '
              'Pre-update coefficients are reconstructed from the initial Adam log variances and '
              'the previous post-update weight snapshot. Curves use the best validation run per mode, '
              'selected once for all loss terms and coefficient plots; seeds appear in legends. These are weighted MSE contributions, '
              'excluding the additive log-variance regularizer in the adaptive objective.', '',
              'Raw values and failed/pending attempts: [runs.csv](runs.csv). '
              'Aggregates: [summary.csv](summary.csv). '
              'Within-seed differences (left minus right): [paired_differences.csv](paired_differences.csv).', '',
              'Only completed finite runs enter aggregates; incomplete counts are explicit. '
              'Each pair uses only seeds completed in both modes. No statistical significance is claimed. '
              'SD is undefined for one run. DE curves use arithmetic means across seeds at each step, '
              'restricted to the common prefix within each mode; no padding or changing cohort. '
              'Shaded bands show sample SD, not confidence intervals. Symmetric-log axes retain nonpositive bounds.', '',
              'FVM is an approximate cell-based reference, compared with pointwise PINN predictions. '
              'Validation and test times are disjoint and never enter training. This is not an exact-solution error. '
              'Initial coefficients and fixed coefficients are recorded in each run metadata; historical studies may use different defaults. '
              'Equal maximum budgets need not yield equal executed budgets or time. '
              'Shared seeds do not make DE populations identical when genome dimensions differ.']
    lines += ['', '![DE convergence](de_convergence.png)', '', '![DE diagnostics](de_diagnostics.png)', '',
              '![PINN loss weights](pinn_loss_weights.png)', '',
              '![PINN fixed reference](pinn_fixed_reference.png)', '',
              'The PINN fixed-reference panels multiply each saved raw training MSE by the '
              'common post-hoc coefficient (Initial=1, Boundary=1, PDE=1). This matches the saved DE reference for de, pinn and both; none uses its recorded fixed coefficients for the original DE reference. The total is '
              'computed from the selected run, without averaging across seeds. This is the '
              'objective along the Adam trajectory, not a best-candidate search or validation error. '
              'All four modes appear in this reference comparison; none is omitted only from '
              'the coefficient plot. DE diagnostics retain the de reference and selected loss '
              'coefficients in the first row. The second row compares de and both: mean F '
              'and mean CR versus generation. Each run first averages over the population; '
              'curves then average those values across seeds. Bands show sample SD across '
              'seed means, not dispersion between individuals. Generation zero is initialization.', '',
              'Plotted values, aggregation type, counts and seeds: [mean_histories.csv](mean_histories.csv).', '',
              'Selected PINN runs: [selected_pinn_runs.json](selected_pinn_runs.json). '
              'Selection minimizes validation RMSE, with seed as the tie-breaker; test error is not used. '
              'These selected trajectories describe best-case runs, not average performance. '
              'Comparison statistics still include every completed finite run.', '',
              'PINN weights are effective loss coefficients, not neural-network parameters. '
              'The de mode freezes DE-selected coefficients during Adam; pinn and both optimize them; '
              'none uses the recorded fixed coefficients. Raw training losses are measured before each update; '
              'Adam weight histories are recorded after each update. DE objectives differ between modes; '
              'use the common fixed-weight reference in the diagnostics for objective comparisons.']
    lines += ['', 'Representative fields place predictions in the first column and absolute errors in the second, with one model per row. The FVM reference occupies its own row. They use the first planned seed and the original training-grid FVM reference; tables use the refined reference. Missing modes are omitted, never replaced by a better seed.']
    lines += [f'![Representative fields]({name})' for name in field_images]
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(f'Report: {output / "README.md"}', flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    analyze(parser.parse_args().directory)
