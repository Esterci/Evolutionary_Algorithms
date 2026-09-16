"""Checks for descriptive statistics and explicit incomplete study records."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.analyze_loss_weight_study import analyze, statistics, MODES, mean_history, weighted_loss_history, select_best_runs


class LossWeightStudyTests(unittest.TestCase):
    def test_best_run_uses_validation_and_seed_tie_break(self):
        completed = [
            (dict(mode='de', seed=2), dict(validation_rmse=.2, test_rmse=.01)),
            (dict(mode='de', seed=1), dict(validation_rmse=.1, test_rmse=10)),
            (dict(mode='pinn', seed=2), dict(validation_rmse=.1, test_rmse=.01)),
            (dict(mode='pinn', seed=1), dict(validation_rmse=.1, test_rmse=10)),
        ]
        self.assertEqual([(r['mode'], r['seed']) for r, _ in select_best_runs(completed)],
                         [('de', 1), ('pinn', 1)])
        self.assertEqual(len(select_best_runs(completed, per_mode=False)), 1)
        self.assertEqual(select_best_runs([]), [])

    def test_weighted_loss_aligns_pre_update_coefficients(self):
        import numpy as np
        weighted = weighted_loss_history([2, 3, 5], [7, 11, 13], 4)
        np.testing.assert_allclose(weighted, [8, 21, 55])
        other = weighted_loss_history([6, 9, 15], [2, 3, 4], 1)
        mean, sd = mean_history([weighted, other])
        np.testing.assert_allclose(mean, [7, 19.5, 50])
        np.testing.assert_allclose(sd, np.std([weighted, other], axis=0, ddof=1))
        for loss, weights in [([1, 2], [1]), ([1], [float('nan')])]:
            with self.assertRaises(ValueError):
                weighted_loss_history(loss, weights, 1)

    def test_curve_mean_uses_common_prefix_and_sample_sd(self):
        import numpy as np
        mean, sd = mean_history([[1, 3, 100], [3, 7]])
        np.testing.assert_allclose(mean, [2, 5])
        np.testing.assert_allclose(sd, [np.sqrt(2), np.sqrt(8)])
        self.assertIsNone(mean_history([[1, 2]])[1])
        with self.assertRaises(ValueError):
            mean_history([[1, float('nan')]])

    def test_adam_epochs_preserve_total_evaluation_budget(self):
        from runners.run_loss_weight_study import adam_epochs_for_budget
        self.assertEqual(adam_epochs_for_budget(8064, 32, 100), 4832)
        self.assertEqual(adam_epochs_for_budget(8064, 32, 200), 1632)
        self.assertEqual(adam_epochs_for_budget(5000, 8, 20), 4832)
        self.assertEqual(adam_epochs_for_budget(10, 4, 1), 2)
        for budget, population, generations in [(8, 4, 1), (7, 4, 1), (10, 3, 1), (10, 4, 0)]:
            with self.subTest(budget=budget, population=population, generations=generations):
                with self.assertRaises(ValueError):
                    adam_epochs_for_budget(budget, population, generations)

    def test_statistics_use_sample_sd_and_do_not_invent_single_run_uncertainty(self):
        self.assertEqual(statistics([])['n'], 0)
        self.assertIsNone(statistics([2])['std'])
        values = statistics([1, 2, 3])
        self.assertEqual(values['mean'], 2)
        self.assertEqual(values['std'], 1)
        self.assertEqual(values['median'], 2)
        self.assertEqual(values['iqr'], 1)
        with self.assertRaises(ValueError):
            statistics([1, float('nan')])

    def test_failed_pending_and_interrupted_attempts_remain_in_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps(dict(runs_per_mode=1,
                generations=1, adam_epochs=2, network_seed_start=1)))
            records = [dict(mode=mode, seed=1, de_seed=2, status=status,
                            error='example' if status == 'failed' else None,
                            log=f'{mode}.log', run=None)
                       for mode, status in zip(MODES, ['failed', 'pending', 'running', 'failed'])]
            (root / 'records.json').write_text(json.dumps(records))
            with contextlib.redirect_stdout(io.StringIO()):
                rows = analyze(root)
            self.assertEqual([r['status'] for r in rows], [r['status'] for r in records])
            summaries = json.loads((root / 'analysis/summary.json').read_text())
            self.assertTrue(all(s['n'] == 0 and s['incomplete'] == 1 and s['mean'] is None for s in summaries))
            self.assertIn('not computed', (root / 'analysis/README.md').read_text())
            self.assertEqual(len((root / 'analysis/paired_differences.csv').read_text().splitlines()), 1)
            self.assertEqual(json.loads((root / 'records.json').read_text()), records)
