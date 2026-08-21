"""
tests/unit/test_experiment_tracker_upgrade.py
─────────────────────────────────────────────
Unit Test Suite for Upgraded Experiment Tracking System.
"""

import unittest
import shutil
import tempfile
from pathlib import Path
import pandas as pd

from ml_layer.experiment_tracker import ExperimentTracker
from alpha_layer.governance.registry import ModelRegistry, load_model


class DummyModel:
    """Mock model to test promotion and serialization."""
    def __init__(self, key="value"):
        self.key = key
        self.features = ["x1", "x2"]


class TestExperimentTrackerUpgrade(unittest.TestCase):

    def setUp(self):
        # Create temp folder for runs and model registry
        self.test_dir = Path(tempfile.mkdtemp())
        self.tracker = ExperimentTracker(tracking_dir=str(self.test_dir / "experiments"))
        self.registry = ModelRegistry(
            registry_path=self.test_dir / "registry.json",
            models_dir=self.test_dir / "models"
        )
        self.exp_name = "governance_experiment"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_run_logging_with_advanced_fields(self):
        # Start a run
        run_id = self.tracker.start_run(
            experiment_name=self.exp_name,
            params={"max_depth": 5, "lr": 0.01},
            tags={"regime": "bull"},
            notes="Testing advanced features"
        )
        self.assertIsNotNone(run_id)

        # Log metrics with the new fields
        metrics = {
            "train_ic": 0.05,
            "val_ic": 0.04,
            "overfit_score": 0.01,
            "decile_sharpe": 1.9,
            "ic_tstat": 2.5,
            "n_train_rows": 1000,
            "n_val_rows": 200,
            "dataset_version": "dataset_v4",
            "feature_version": "features_v2",
            "sharpe": 1.75,
            "drawdown": 0.11,
            "cost_drag": 14.5,
            "feature_importance": {"momentum": 0.6, "value": 0.4}
        }
        self.tracker.log_metrics(run_id, self.exp_name, metrics, elapsed_seconds=12.4)

        # Query runs and check fields
        df = self.tracker.get_runs(self.exp_name)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["run_id"], run_id)
        self.assertEqual(row["dataset_version"], "dataset_v4")
        self.assertEqual(row["feature_version"], "features_v2")
        self.assertEqual(row["sharpe"], 1.75)
        self.assertEqual(row["drawdown"], 0.11)
        self.assertEqual(row["cost_drag"], 14.5)
        self.assertEqual(row["feature_importance"]["momentum"], 0.6)

    def test_run_comparison(self):
        # Create Run A
        run_a = self.tracker.start_run(self.exp_name, params={"depth": 3})
        self.tracker.log_metrics(run_a, self.exp_name, {
            "val_ic": 0.03, "train_ic": 0.04, "sharpe": 1.2, "drawdown": 0.15, "cost_drag": 10.0,
            "feature_importance": {"f1": 0.8}
        })

        # Create Run B
        run_b = self.tracker.start_run(self.exp_name, params={"depth": 5})
        self.tracker.log_metrics(run_b, self.exp_name, {
            "val_ic": 0.05, "train_ic": 0.06, "sharpe": 1.8, "drawdown": 0.10, "cost_drag": 12.0,
            "feature_importance": {"f1": 0.9}
        })

        # Compare
        diff = self.tracker.compare_runs(self.exp_name, run_a, run_b)
        self.assertAlmostEqual(diff["metrics_diff"]["sharpe_diff"], 0.6)
        self.assertAlmostEqual(diff["metrics_diff"]["val_ic_diff"], 0.02)
        self.assertEqual(diff["param_diff"]["depth"]["run_a"], 3)
        self.assertEqual(diff["param_diff"]["depth"]["run_b"], 5)
        self.assertEqual(diff["feat_diff"]["f1"]["run_a"], 0.8)
        self.assertEqual(diff["feat_diff"]["f1"]["run_b"], 0.9)

    def test_generate_report(self):
        run_id = self.tracker.start_run(self.exp_name, params={"lr": 0.05})
        self.tracker.log_metrics(run_id, self.exp_name, {
            "val_ic": 0.04, "sharpe": 1.5, "drawdown": 0.12, "cost_drag": 15.0,
            "feature_importance": {"feat1": 0.7, "feat2": 0.3}
        })

        # Generate report content
        report_md = self.tracker.generate_report(self.exp_name, run_id)
        self.assertIn("QuantSphereX Experiment Report", report_md)
        self.assertIn("Validation IC**: 0.0400", report_md)
        self.assertIn("Net Sharpe Ratio**: 1.50", report_md)
        self.assertIn("lr", report_md)

        # Write to file
        report_path = self.test_dir / "report.md"
        self.tracker.generate_report(self.exp_name, run_id, output_path=str(report_path))
        self.assertTrue(report_path.exists())
        self.assertIn("Validation IC", report_path.read_text(encoding="utf-8"))

    def test_promotion_to_model_registry(self):
        run_id = self.tracker.start_run(self.exp_name, params={"depth": 4})
        self.tracker.log_metrics(run_id, self.exp_name, {
            "val_ic": 0.045, "train_ic": 0.055, "sharpe": 1.6, "drawdown": 0.10, "cost_drag": 8.0,
            "dataset_version": "ds_promo", "feature_version": "feat_promo"
        })

        model = DummyModel(key="promoted_test")

        # Promote run to model registry
        saved_path = self.tracker.promote_to_registry(
            experiment_name=self.exp_name,
            run_id=run_id,
            model_object=model,
            model_registry=self.registry,
            model_id="PromotedModel",
            version="1.0.0",
            status="Production",
            notes="Promotion testing"
        )
        self.assertTrue(saved_path.exists())

        # Verify registry record
        loaded_model, loaded_meta = load_model("PromotedModel", "1.0.0", registry=self.registry)
        self.assertEqual(loaded_meta.model_id, "PromotedModel")
        self.assertEqual(loaded_meta.version, "1.0.0")
        self.assertEqual(loaded_meta.status, "Production")
        self.assertEqual(loaded_meta.sharpe, 1.6)
        self.assertEqual(loaded_meta.dataset_version, "ds_promo")
        self.assertEqual(loaded_model.key, "promoted_test")

        # Verify that the tracker record was updated with the model info
        run_record = self.tracker._load_run(run_id, self.exp_name)
        self.assertEqual(run_record.model_id, "PromotedModel")
        self.assertEqual(run_record.model_version, "1.0.0")
