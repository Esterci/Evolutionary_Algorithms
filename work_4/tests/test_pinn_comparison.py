"""Check saved-run compatibility, metrics and multi-method visualization."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_pinn_results import (
    comparison_panel,
    comparison_report,
    load_run,
    main,
    plot_histories,
    select_runs,
    validate_runs,
    plt,
)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directories = []
        for name, error, hybrid in [("adam", 1.0, False), ("hybrid", 0.5, True)]:
            run = self.root / name
            run.mkdir()
            self.directories.append(run)
            reference = np.ones((2, 2, 2))
            np.savez(
                run / "comparison.npz",
                x=[0.25, 0.75],
                y=[0.25, 0.75],
                t=[0.0, 0.1],
                u=reference + error,
                v=reference - error,
                fvm_u=reference,
                fvm_v=reference,
            )
            metadata = dict(
                optimizer="DE -> Adam" if hybrid else "Adam",
                mesh={"h": 0.5},
                constants={"nu": 0.01},
                adaptive_loss=False,
                epochs=4,
            )
            if hybrid:
                metadata.update(
                    de_generations=2, adam_epochs=2, de={"fitness_evaluations": 12}
                )
                np.savez(
                    run / "de_history.npz",
                    fitness_evaluations=[4, 8, 12],
                    best_fitness=[4, 2, 1],
                    mean_fitness=[5, 3, 2],
                )
            (run / "metadata.json").write_text(json.dumps(metadata))
            np.savez(run / "losses.npz", Initial=[2.0, 1.0], PDE=[0.2, 0.1])
        self.runs = [load_run(path) for path in self.directories]

    def test_selection_and_metrics_identify_both_methods(self):
        self.assertEqual(set(select_runs(self.root)), set(self.directories))
        validate_runs(self.runs)
        rows = comparison_report(self.runs)
        self.assertEqual([r["rmse"] for r in rows], [1.0, 1.0, 0.5, 0.5])
        self.assertEqual([r["relative_l2"] for r in rows], [1.0, 1.0, 0.5, 0.5])
        self.assertEqual(rows[2]["fitness_evaluations"], 12)
        self.assertIn("DE", rows[2]["method"])
        self.runs[0]["arrays"]["fvm_u"].fill(0)
        self.assertIsNone(comparison_report(self.runs)[0]["relative_l2"])

    def test_rejects_incompatible_physics_grid_and_reference(self):
        for key in ("mesh", "constants"):
            original = self.runs[1]["metadata"][key]
            self.runs[1]["metadata"][key] = {"different": True}
            with self.assertRaisesRegex(ValueError, key):
                validate_runs(self.runs)
            self.runs[1]["metadata"][key] = original
        for key in ("x", "t", "fvm_u"):
            self.runs[1]["arrays"][key] += 0.01
            with self.assertRaisesRegex(ValueError, key):
                validate_runs(self.runs)
            self.runs[1]["arrays"][key] -= 0.01

    def test_shared_field_and_error_scales(self):
        figure, update = comparison_panel(
            self.runs[0]["arrays"], {r["label"]: r["arrays"] for r in self.runs}
        )
        self.addCleanup(plt.close, figure)
        meshes = [a.collections[0] for a in figure.axes if a.get_xlabel() == "x"]
        self.assertEqual(len(meshes), 10)
        self.assertIs(meshes[0].norm, meshes[1].norm)
        self.assertIs(meshes[1].norm, meshes[2].norm)
        self.assertIs(meshes[3].norm, meshes[4].norm)
        update(1)

    def test_legacy_single_run_without_metadata_still_loads_and_plots(self):
        (self.directories[0] / "metadata.json").unlink()
        run = load_run(self.directories[0])
        validate_runs([run])
        self.assertIn("unknown", run["label"])
        figure, update = comparison_panel(run["arrays"])
        self.addCleanup(plt.close, figure)
        self.assertEqual(len(update(0)), 7)

    def test_cli_writes_combined_reports_figures_and_de_history(self):
        output = self.root / "plots"
        with contextlib.redirect_stdout(io.StringIO()):
            main(
                [
                    "--run-directory",
                    str(self.directories[0]),
                    "--run-directory",
                    str(self.directories[1]),
                    "--output-directory",
                    str(output),
                    "--frames",
                    "2",
                    "--fps",
                    "2",
                ]
            )
        for name in (
            "comparison_metrics.json",
            "comparison_metrics.csv",
            "comparison_time_00000.png",
            "pinn_fvm_comparison.gif",
            "error_evolution.png",
            "training_losses.png",
            "de_convergence_hybrid.png",
        ):
            self.assertGreater((output / name).stat().st_size, 0)
        report = json.loads((output / "comparison_metrics.json").read_text())
        self.assertEqual(len(report["metrics"]), 4)
        self.assertEqual(len(report["runs"]), 2)

    def test_regularized_negative_fitness_is_plotted_without_log_clipping(self):
        run = self.runs[1]
        run["metadata"]["de_adaptive_pde_weights"] = True
        np.savez(
            run["directory"] / "de_history.npz",
            fitness_evaluations=[4, 8, 12],
            best_fitness=[1.0, 0.0, -1.0],
            mean_fitness=[2.0, 1.0, -0.5],
            selected_reference_fitness=[3.0, 4.0, 2.0],
            best_reference_fitness=[3.0, 3.0, 2.0],
            best_weight_PDE_u=[0.5, 0.4, 0.3],
            best_weight_PDE_v=[0.5, 0.6, 0.7],
        )
        output = self.root / "negative_plots"
        output.mkdir()
        captured = []
        original_subplots = plt.subplots

        def capture(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            captured.append((figure, axes))
            return figure, axes

        with patch("plot_pinn_results.plt.subplots", side_effect=capture):
            plot_histories([run], output)
        convergence = captured[1][1]
        self.assertEqual(convergence.get_yscale(), "symlog")
        np.testing.assert_array_equal(
            convergence.lines[0].get_ydata(), [1.0, 0.0, -1.0]
        )
        self.assertTrue((output / "de_diagnostics_hybrid.png").is_file())

    def test_three_adaptive_weights_are_reported_and_plotted_through_both_stages(self):
        run = self.runs[1]
        names = ["Initial", "Boundary", "PDE"]
        run["metadata"].update(
            adaptive_loss=True,
            de_adaptive_loss_weights=True,
            loss_weight_names=names,
            adaptive_formula="sum_i(exp(-s_i)*L_i + s_i)",
            de_best_loss_weights=dict(zip(names, [0.3, 0.4, 0.5])),
            final_loss_weights=dict(zip(names, [0.2, 0.3, 0.4])),
            adam_loss_weighting="adaptive; initialized from the DE-selected log variances",
        )
        np.savez(
            run["directory"] / "de_history.npz",
            fitness_evaluations=[4, 8, 12],
            best_fitness=[1.0, 0.0, -1.0],
            mean_fitness=[2.0, 1.0, -0.5],
            selected_reference_fitness=[3.0, 4.0, 2.0],
            best_reference_fitness=[3.0, 3.0, 2.0],
            **{f"best_weight_{name}": [0.5, 0.4, 0.3] for name in names},
        )
        weights = np.array([[0.3, 0.4, 0.5], [0.2, 0.3, 0.4]])
        np.savez(
            run["directory"] / "adam_loss_weights.npz",
            loss_names=names,
            weights=weights,
            log_vars=-np.log(weights),
        )
        output = self.root / "adaptive_plots"
        output.mkdir()
        captured = []
        original_subplots = plt.subplots

        def capture(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            captured.append((figure, axes))
            return figure, axes

        with patch("plot_pinn_results.plt.subplots", side_effect=capture):
            plot_histories([run], output)
        adam_axis, convergence_axis, de_axes = [entry[1] for entry in captured[1:]]
        self.assertEqual([line.get_label() for line in adam_axis.lines], names)
        np.testing.assert_array_equal(adam_axis.lines[0].get_xdata(), [1, 2])
        np.testing.assert_array_equal(adam_axis.lines[0].get_ydata(), weights[:, 0])
        self.assertEqual(convergence_axis.get_ylabel(), "Regularized PINN fitness")
        self.assertEqual([line.get_label() for line in de_axes[1].lines], names)
        self.assertTrue((output / "adam_loss_weights_hybrid.png").is_file())

        current = comparison_report([run])[0]
        self.assertTrue(current["adaptive_loss"])
        self.assertTrue(current["de_adaptive_loss_weights"])
        self.assertFalse(current["de_adaptive_pde_weights"])
        self.assertEqual(current["loss_weight_names"], names)
        self.assertEqual(current["final_loss_weights"]["Initial"], 0.2)
        self.assertEqual(current["de_best_loss_weights"]["Initial"], 0.3)
        self.assertEqual(current["adaptive_formula"], run["metadata"]["adaptive_formula"])

    def test_legacy_pde_only_adaptation_is_not_reported_as_three_term_adaptation(self):
        run = self.runs[1]
        run["metadata"].update(
            adaptive_loss=False,
            de_adaptive_pde_weights=True,
            adam_loss_weighting="fixed coefficients inherited from DE",
        )
        legacy = comparison_report([run])[0]
        self.assertFalse(legacy["adaptive_loss"])
        self.assertFalse(legacy["de_adaptive_loss_weights"])
        self.assertTrue(legacy["de_adaptive_pde_weights"])
        self.assertIsNone(legacy["adaptive_formula"])


if __name__ == "__main__":
    unittest.main()
