"""
tests/unit/test_confidence.py
─────────────────────────────
Unit tests for ml_layer/confidence.py (Confidence Estimation Module).
"""

import unittest
import numpy as np
import pandas as pd
from alpha_layer.xgboost_trainer import EnsembleAlphaModel, XGBoostAlphaModel
from ml_layer.confidence import (
    ensemble_variance,
    confidence_tiers,
    conformal_intervals,
    full_confidence_report,
)


class TestConfidenceModule(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=20)
        self.X = pd.DataFrame(
            np.random.randn(20, 4),
            index=dates,
            columns=["f1", "f2", "f3", "f4"],
        )
        self.y = pd.Series(np.random.randn(20), index=dates)

        # Build mock ensemble with 2 models
        self.ensemble = EnsembleAlphaModel(n_models=2)
        m1 = XGBoostAlphaModel()
        m2 = XGBoostAlphaModel()
        m1.fit(self.X, self.y)
        m2.fit(self.X, self.y + np.random.normal(0, 0.1, len(self.y)))
        self.ensemble.models = [m1, m2]

    def test_ensemble_variance(self):
        var_series = ensemble_variance(self.ensemble, self.X)
        self.assertEqual(len(var_series), len(self.X))
        self.assertTrue((var_series >= 0).all())

    def test_confidence_tiers(self):
        tiers = confidence_tiers(self.ensemble, self.X)
        self.assertEqual(len(tiers), len(self.X))
        self.assertTrue(set(tiers.unique()).issubset({"HIGH", "MEDIUM", "LOW"}))

    def test_conformal_intervals(self):
        intervals = conformal_intervals(
            self.ensemble,
            X_cal=self.X.head(10),
            y_cal=self.y.head(10),
            X_test=self.X.tail(10),
            coverage=0.90,
        )
        self.assertEqual(len(intervals), 10)
        self.assertIn("predicted", intervals.columns)
        self.assertIn("lower_bound", intervals.columns)
        self.assertIn("upper_bound", intervals.columns)
        self.assertTrue((intervals["upper_bound"] >= intervals["lower_bound"]).all())

    def test_full_confidence_report(self):
        report = full_confidence_report(
            self.ensemble,
            X=self.X,
            X_cal=self.X.head(10),
            y_cal=self.y.head(10),
        )
        self.assertEqual(len(report), len(self.X))
        self.assertIn("alpha_score", report.columns)
        self.assertIn("prediction_std", report.columns)
        self.assertIn("confidence_tier", report.columns)
        self.assertIn("lower_90", report.columns)
        self.assertIn("upper_90", report.columns)


if __name__ == "__main__":
    unittest.main()
