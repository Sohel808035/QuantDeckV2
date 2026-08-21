"""
tests/unit/test_governance.py
──────────────────────────────
Unit Test Suite for Model Governance System.
"""

import os
import unittest
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from alpha_layer.governance.registry import (
    ModelRegistry,
    ModelMetadata,
    save_model,
    load_model,
)
from alpha_layer.governance.comparison import compare_models
from alpha_layer.governance.reproducibility import verify_reproducibility
from alpha_layer.xgboost_trainer import EnsembleAlphaModel


class DummyModel:
    """Mock model to test serialization and prediction."""
    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier
        self.features = ["feat1", "feat2"]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Sum features and multiply
        return (X["feat1"] + X["feat2"]) * self.multiplier


class TestModelGovernance(unittest.TestCase):

    def setUp(self):
        # Create a temp directory for models and registry
        self.test_dir = Path(tempfile.mkdtemp())
        self.registry_path = self.test_dir / "registry.json"
        self.models_dir = self.test_dir / "models"
        
        self.registry = ModelRegistry(
            registry_path=self.registry_path,
            models_dir=self.models_dir
        )

        # Generate some mock data for reproducibility test
        np.random.seed(42)
        self.mock_X = pd.DataFrame({
            "feat1": np.random.randn(100),
            "feat2": np.random.randn(100)
        })

    def tearDown(self):
        # Clean up temp folder
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_metadata_validation(self):
        # Valid metadata
        meta = ModelMetadata(
            model_id="TestModel",
            version="1.0.0",
            dataset_version="ds_v1",
            feature_version="feat_v1",
            hyperparameters={"param1": 10},
            validation_ic=0.04,
            train_ic=0.06,
            sharpe=1.8,
            drawdown=0.12,
            status="Production",
            git_commit_hash="abc12345",
            notes="Initial production model"
        )
        self.assertEqual(meta.model_id, "TestModel")
        self.assertEqual(meta.status, "Production")

        # Invalid status
        with self.assertRaises(ValueError):
            ModelMetadata(
                model_id="TestModel",
                version="1.0.0",
                dataset_version="ds_v1",
                feature_version="feat_v1",
                hyperparameters={},
                validation_ic=0.0,
                train_ic=0.0,
                sharpe=0.0,
                drawdown=0.0,
                status="InvalidStatus",
                git_commit_hash="abc"
            )

        # Invalid version format
        with self.assertRaises(ValueError):
            ModelMetadata(
                model_id="TestModel",
                version="invalid_version",
                dataset_version="ds_v1",
                feature_version="feat_v1",
                hyperparameters={},
                validation_ic=0.0,
                train_ic=0.0,
                sharpe=0.0,
                drawdown=0.0,
                status="Experimental",
                git_commit_hash="abc"
            )

    def test_save_and_load_model(self):
        model = DummyModel(multiplier=2.5)
        meta_args = {
            "model_id": "DummyEnsemble",
            "version": "1.0.0",
            "dataset_version": "ds_v1",
            "feature_version": "feat_v1",
            "hyperparameters": {"multiplier": 2.5},
            "validation_ic": 0.05,
            "train_ic": 0.08,
            "sharpe": 2.1,
            "drawdown": 0.08,
            "status": "Experimental",
            "git_commit_hash": "unknown",
            "notes": "Test saving dummy"
        }

        # Save model
        saved_path = save_model(model, meta_args, registry=self.registry)
        self.assertTrue(saved_path.exists())
        self.assertEqual(saved_path.name, "DummyEnsemble_1.0.0.pkl")

        # Load model
        loaded_model, loaded_meta = load_model("DummyEnsemble", "1.0.0", registry=self.registry)
        self.assertEqual(loaded_meta.model_id, "DummyEnsemble")
        self.assertEqual(loaded_model.multiplier, 2.5)

        # Check prediction equivalence
        preds_orig = model.predict(self.mock_X)
        preds_loaded = loaded_model.predict(self.mock_X)
        pd.testing.assert_series_equal(preds_orig, preds_loaded)

    def test_rollback_on_failure(self):
        # 1. Save a Production model version 1.0.0
        model_prod = DummyModel(multiplier=1.0)
        meta_prod = {
            "model_id": "RollbackModel",
            "version": "1.0.0",
            "dataset_version": "ds_v1",
            "feature_version": "feat_v1",
            "hyperparameters": {"multiplier": 1.0},
            "validation_ic": 0.04,
            "train_ic": 0.05,
            "sharpe": 1.5,
            "drawdown": 0.10,
            "status": "Production",
            "git_commit_hash": "unknown"
        }
        save_model(model_prod, meta_prod, registry=self.registry)

        # 2. Save an Experimental model version 2.0.0
        model_exp = DummyModel(multiplier=2.0)
        meta_exp = {
            "model_id": "RollbackModel",
            "version": "2.0.0",
            "dataset_version": "ds_v2",
            "feature_version": "feat_v1",
            "hyperparameters": {"multiplier": 2.0},
            "validation_ic": 0.01, # Poor IC
            "train_ic": 0.06,
            "sharpe": 0.8,
            "drawdown": 0.22,
            "status": "Experimental",
            "git_commit_hash": "unknown"
        }
        save_model(model_exp, meta_exp, registry=self.registry)

        # Define validation function that fails for version 2.0.0 (e.g. requires validation_ic > 0.02)
        def validate_performance(model, meta):
            return meta.validation_ic >= 0.02

        # 3. Load version 2.0.0 with verification -> should trigger rollback to 1.0.0
        loaded_model, loaded_meta = load_model(
            "RollbackModel",
            "2.0.0",
            registry=self.registry,
            validation_fn=validate_performance,
            fallback_on_failure=True
        )

        # Assert it rolled back to version 1.0.0
        self.assertEqual(loaded_meta.version, "1.0.0")
        self.assertEqual(loaded_model.multiplier, 1.0)

    def test_compare_models(self):
        m1 = DummyModel(multiplier=1.0)
        m2 = DummyModel(multiplier=2.0)

        meta_1 = {
            "model_id": "CompModel",
            "version": "1.0.0",
            "dataset_version": "ds_1",
            "feature_version": "feat_1",
            "hyperparameters": {"multiplier": 1.0},
            "validation_ic": 0.04,
            "train_ic": 0.05,
            "sharpe": 1.4,
            "drawdown": 0.15,
            "status": "Production",
            "git_commit_hash": "hash1"
        }
        meta_2 = {
            "model_id": "CompModel",
            "version": "2.0.0",
            "dataset_version": "ds_2",
            "feature_version": "feat_1",
            "hyperparameters": {"multiplier": 2.0},
            "validation_ic": 0.06,
            "train_ic": 0.08,
            "sharpe": 1.9,
            "drawdown": 0.10,
            "status": "Experimental",
            "git_commit_hash": "hash2"
        }

        save_model(m1, meta_1, registry=self.registry)
        save_model(m2, meta_2, registry=self.registry)

        report = compare_models("CompModel", "1.0.0", "2.0.0", registry=self.registry)

        self.assertEqual(report["metrics_diff"]["sharpe_diff"], 0.5)
        self.assertAlmostEqual(report["metrics_diff"]["drawdown_diff"], -0.05)
        self.assertIn("multiplier", report["hyperparameter_diff"])

    def test_verify_reproducibility(self):
        m1 = DummyModel(multiplier=1.5)
        m2 = DummyModel(multiplier=1.5)
        m_diff = DummyModel(multiplier=1.6)

        # Same model parameters should pass reproducibility verification
        res_pass = verify_reproducibility(m1, m2, self.mock_X)
        self.assertTrue(res_pass)

        # Different model parameters should fail reproducibility verification
        res_fail = verify_reproducibility(m1, m_diff, self.mock_X)
        self.assertFalse(res_fail)
