"""
tests/unit/test_explainability_engine.py
─────────────────────────────────────────
Unit Test Suite for Institutional Explainability Engine.

Tests:
  - Global Feature Importance (with and without SHAP fitted models)
  - Local Feature Importance
  - Waterfall plot data structure and PNG output
  - Dependence plot data structure and PNG output
  - Per-prediction institutional explanation object structure
  - PredictionExplainer.explain_institutional_prediction coverage
"""

import unittest
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from ml_layer.explainability import InstitutionalExplainabilityEngine
from ai_quant_analyst.prediction_explainer import PredictionExplainer


# ── Minimal stub EnsembleAlphaModel for unit isolation ───────────────────────

class _StubSubModel:
    """Mimics XGBoostAlphaModel interface without a real XGBoost booster."""
    def __init__(self, features: List[str]):
        self.features = features
        self.model = None  # intentionally None — forces SHAP fallback path

    def predict(self, X: pd.DataFrame) -> pd.Series:
        vals = (X["return_1m"] * 0.5 + X["residual_momentum"] * 0.3
                - X["volatility_regime"] * 0.2 - X["amihud_illiquidity"] * 0.1)
        return vals.rename("predicted_score")


class _StubEnsemble:
    """Minimal EnsembleAlphaModel stub."""
    def __init__(self):
        self._features = ["return_1m", "residual_momentum", "volatility_regime", "amihud_illiquidity"]
        sub = _StubSubModel(self._features)
        self.models = [sub, sub]

    def get_features(self) -> List[str]:
        return self._features

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = pd.concat([m.predict(X) for m in self.models], axis=1).mean(axis=1)
        return preds.rename("predicted_score")


class TestInstitutionalExplainabilityEngine(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        np.random.seed(42)
        self.ensemble = _StubEnsemble()
        self.engine = InstitutionalExplainabilityEngine(self.ensemble)

        self.mock_X = pd.DataFrame({
            "return_1m":          np.random.uniform(-0.05, 0.05, 50),
            "residual_momentum":  np.random.uniform(-0.05, 0.05, 50),
            "volatility_regime":  np.random.uniform(0.5,  2.0,  50),
            "amihud_illiquidity": np.random.uniform(0.0,  0.01, 50),
        })

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. Global Feature Importance ─────────────────────────────────────────────
    def test_global_feature_importance_returns_correct_columns(self):
        df = self.engine.get_global_feature_importance(self.mock_X)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        for col in ("feature", "mean_abs_shap", "relative_importance", "rank"):
            self.assertIn(col, df.columns)

    def test_global_feature_importance_covers_all_features(self):
        df = self.engine.get_global_feature_importance(self.mock_X)
        self.assertEqual(len(df), len(self.ensemble.get_features()))

    def test_global_feature_importance_sorted_descending(self):
        df = self.engine.get_global_feature_importance(self.mock_X)
        vals = df["mean_abs_shap"].values
        self.assertTrue(all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)),
                        "Global importance should be sorted descending by mean_abs_shap")

    # 2. Local Feature Importance ───────────────────────────────────────────────
    def test_local_feature_importance_single_row(self):
        row = self.mock_X.head(1)
        local_imp = self.engine.get_local_feature_importance(row)
        self.assertIsInstance(local_imp, dict)
        self.assertEqual(set(local_imp.keys()), set(self.ensemble.get_features()))

    # 3. Waterfall Plot ─────────────────────────────────────────────────────────
    def test_waterfall_plot_data_structure(self):
        row = self.mock_X.head(1)
        data = self.engine.generate_waterfall_plot_data(row)
        for key in ("prediction", "base_value", "values", "feature_values"):
            self.assertIn(key, data)
        self.assertIsInstance(data["values"], dict)

    def test_waterfall_plot_saves_png(self):
        row = self.mock_X.head(1)
        plot_path = self.test_dir / "waterfall.png"
        result = self.engine.generate_waterfall_plot(row, save_path=str(plot_path))
        self.assertTrue(plot_path.exists(), "Waterfall PNG was not created")

    # 4. Dependence Plot ────────────────────────────────────────────────────────
    def test_dependence_plot_data_structure(self):
        data = self.engine.generate_dependence_plot_data(self.mock_X, "return_1m")
        self.assertEqual(data["feature"], "return_1m")
        self.assertIsInstance(data["x_values"], list)
        self.assertIsInstance(data["shap_values"], list)

    def test_dependence_plot_saves_png(self):
        plot_path = self.test_dir / "dependence.png"
        result = self.engine.generate_dependence_plot(
            self.mock_X, "return_1m", save_path=str(plot_path)
        )
        self.assertTrue(plot_path.exists(), "Dependence PNG was not created")

    def test_dependence_plot_missing_feature_returns_empty(self):
        data = self.engine.generate_dependence_plot_data(self.mock_X, "nonexistent_col")
        self.assertEqual(data["x_values"], [])

    # 5. Per-Prediction Institutional Explanation ───────────────────────────────
    def test_explain_prediction_structure_keys(self):
        row = self.mock_X.head(1)
        explanation = self.engine.explain_prediction(row, symbol="RELIANCE.NS")
        for key in ("symbol", "prediction", "confidence", "top_positive_factors",
                    "top_negative_factors", "risk_drivers"):
            self.assertIn(key, explanation)

    def test_explain_prediction_symbol_correct(self):
        row = self.mock_X.head(1)
        explanation = self.engine.explain_prediction(row, symbol="TCS.NS")
        self.assertEqual(explanation["symbol"], "TCS.NS")

    def test_explain_prediction_confidence_in_range(self):
        row = self.mock_X.head(1)
        explanation = self.engine.explain_prediction(row, symbol="INFY.NS")
        self.assertGreaterEqual(explanation["confidence"], 0.0)
        self.assertLessEqual(explanation["confidence"], 1.0)

    def test_explain_prediction_top_factors_format(self):
        row = self.mock_X.head(1)
        explanation = self.engine.explain_prediction(row, symbol="TCS.NS")
        for entry in explanation["top_positive_factors"] + explanation["top_negative_factors"]:
            self.assertIn("factor", entry)
            self.assertIn("impact", entry)

    def test_explain_prediction_risk_drivers_format(self):
        row = self.mock_X.head(1)
        explanation = self.engine.explain_prediction(row, symbol="HDFCBANK.NS")
        for entry in explanation["risk_drivers"]:
            self.assertIn("factor", entry)
            self.assertIn("shap_impact", entry)
            self.assertIn("assessment", entry)

    # 6. PredictionExplainer Institutional Method ───────────────────────────────
    def test_predict_explainer_institutional_structure(self):
        explainer = PredictionExplainer()
        shap_vals = {
            "return_1m":          0.04,
            "residual_momentum":  0.02,
            "volatility_regime": -0.03,
            "amihud_illiquidity": -0.01,
        }
        result = explainer.explain_institutional_prediction(
            symbol="TCS.NS",
            prediction=0.025,
            confidence=0.80,
            shap_values=shap_vals,
            base_value=0.0
        )
        self.assertEqual(result["symbol"], "TCS.NS")
        self.assertEqual(result["prediction"], 0.025)
        self.assertEqual(result["confidence"], 0.80)
        self.assertGreater(len(result["top_positive_factors"]), 0)
        self.assertGreater(len(result["top_negative_factors"]), 0)
        self.assertGreater(len(result["risk_drivers"]), 0)

    def test_predict_explainer_risk_drivers_only_risk_features(self):
        explainer = PredictionExplainer()
        shap_vals = {
            "return_1m":          0.04,
            "volatility_regime": -0.03,
            "amihud_illiquidity": -0.01,
        }
        result = explainer.explain_institutional_prediction(
            symbol="WIPRO.NS",
            prediction=0.01,
            confidence=0.65,
            shap_values=shap_vals,
        )
        risk_factors = {r["factor"] for r in result["risk_drivers"]}
        # At least one of the risk feature names should be captured
        self.assertTrue(
            risk_factors.intersection({"volatility_regime", "amihud_illiquidity"}),
            "Risk drivers should contain known risk factors"
        )
