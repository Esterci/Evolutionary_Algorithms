"""Build a standalone report from this experiment's six runs only."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def markdown(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def statistic(row, key, digits=6):
    return f"{row[key + '_mean']:.{digits}f} ± {row[key + '_std']:.{digits}f}"


records = read(ROOT / 'confirm_records.json')
summaries = read(ROOT / 'confirmation_summary.json')
assert len(records) == 6 and all(r['returncode'] == 0 and 'evaluation' in r for r in records)
assert all(r['runs'] == 3 and r['failed_runs'] == 0 for r in summaries)
labels = {'pinn': 'PINN', 'pinn-de': 'PINN-DE'}
results, times = [], []
for row in summaries:
    results.append([labels[row['method']]] + [statistic(row, k) for k in
                    ('validation', 'test', 'physics_joint_rmse', 'initial_rmse')])
    times.append([labels[row['method']]] + [statistic(row, k, 1) for k in
                  ('de_seconds', 'adam_seconds', 'training_seconds')])
seeds = []
for seed in (2028, 2029, 2030):
    pair = {r['method']: r['evaluation'] for r in records if r['seed'] == seed}
    a, b = pair['pinn'], pair['pinn-de']
    assert b['test']['joint_rmse'] < a['test']['joint_rmse']
    seeds.append([seed, f"{a['test']['joint_rmse']:.6f}", f"{b['test']['joint_rmse']:.6f}",
                  f"{a['physics_joint_rmse']:.6f}", f"{b['physics_joint_rmse']:.6f}"])
diagnostics = []
for record in records:
    if record['method'] != 'pinn-de':
        continue
    run = Path(record['run'])
    metadata = read(run / 'metadata.json')
    with (run / 'de_history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    diagnostics.append(dict(seed=record['seed'], individuals=metadata['de']['population_size'],
        generations=metadata['de']['generations'], initial_best_fitness=float(history[0]['best_fitness']),
        final_best_fitness=float(history[-1]['best_fitness']),
        accepted_trials=sum(int(h['accepted_trials']) for h in history),
        accepted_control_proposals=sum(int(h['control_adaptation_acceptances']) for h in history)))
with (ROOT / 'experiment_de_diagnostics.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
    writer.writeheader()
    writer.writerows(diagnostics)
de_rows = [[r['seed'], r['individuals'], r['generations'], f"{r['initial_best_fitness']:.6f}",
            f"{r['final_best_fitness']:.6f}", r['accepted_trials'], r['accepted_control_proposals']]
           for r in diagnostics]
objective = read(ROOT / 'constant_field_objective_check.json')
objective_rows = [[f"{-r['log_vars'][1]:g}", 0.25, 0, 0, f"{r['adaptive_objective']:g}"]
                  for r in objective['cases']]
sections = dict(
    RESULTS=markdown(['Method', 'Validation RMSE', 'Test RMSE', 'PDE RMSE', 'Initial RMSE'], results),
    TIMES=markdown(['Method', 'DE time (s)', 'Adam time (s)', 'Total training time (s)'], times),
    SEEDS=markdown(['Seed', 'PINN test RMSE', 'PINN-DE test RMSE', 'PINN PDE RMSE', 'PINN-DE PDE RMSE'], seeds),
    DE=markdown(['Seed', 'Individuals', 'Generations', 'Initial best fitness', 'Final best fitness',
                 'Accepted trials', 'Accepted control proposals'], de_rows),
    OBJECTIVE=markdown(['k', 'Initial MSE', 'Boundary MSE', 'PDE MSE', 'Adaptive objective'], objective_rows),
)
text = (ROOT / 'report_context.md').read_text()
for key, value in sections.items():
    text = text.replace('{{' + key + '}}', value)
assert '{{' not in text
(ROOT / 'README.md').write_text(text)
print('Standalone report:', ROOT / 'README.md')
