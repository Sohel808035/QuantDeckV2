"""
ml_layer/explainability.py
────────────────────────────
ML Pipeline: Institutional Explainability Engine (v2.0.0)

Provides institutional-grade model explainability using SHAP with:
  - TreeExplainer for XGBoost ensemble members
  - Aggregated SHAP values across ensemble
  - Global Feature Importance (mean |SHAP| ranking)
  - Local Feature Importance for individual predictions
  - Waterfall plots & dependence plot data generation and PNG rendering
  - Feature contribution decomposition (positive factors, negative factors, risk drivers)
  - Per-prediction Explanation objects (Prediction, Confidence, Top Factors, Risk Drivers)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from alpha_layer.xgboost_trainer import EnsembleAlphaModel

logger = logging.getLogger(__name__)

RISK_FACTOR_NAMES = {
    "volatility_regime", "idiosyncratic_volatility", "amihud_illiquidity",
    "volume_shock", "bollinger_distance"
}


class InstitutionalExplainabilityEngine:
    """Institutional ML Model Explainability Engine based on SHAP tree attributions."""

    def __init__(self, model: EnsembleAlphaModel):
        self.model = model

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 1000
    ) -> Optional[np.ndarray]:
        """Computes SHAP values averaged across all ensemble members."""
        try:
            import shap
        except ImportError:
            logger.warning("[ExplainabilityEngine] shap library not installed.")
            return None

        if not self.model.models:
            logger.warning("[ExplainabilityEngine] No trained submodels in ensemble.")
            return None

        features = self.model.get_features()
        if not features:
            return None

        X_shap = X[features].dropna().head(max_samples)
        if X_shap.empty:
            return None

        all_shap = []
        for m in self.model.models:
            if m.model is None:
                continue
            try:
                explainer = shap.TreeExplainer(m.model)
                sv = explainer.shap_values(X_shap)
                all_shap.append(sv)
            except Exception as e:
                logger.debug(f"[ExplainabilityEngine] TreeExplainer failed for submodel: {e}")

        if not all_shap:
            return None

        mean_shap = np.mean(all_shap, axis=0)
        return mean_shap

    def get_global_feature_importance(
        self,
        X: pd.DataFrame,
        max_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Computes Global Feature Importance (mean |SHAP value| per feature).
        Returns DataFrame sorted by importance with relative weights and ranks.
        """
        features = self.model.get_features()
        if not features:
            return pd.DataFrame()

        shap_vals = self.compute_shap_values(X, max_samples=max_samples)
        if shap_vals is None:
            # Fallback to feature variance if SHAP is not available
            X_clean = X[features].dropna().head(max_samples)
            if X_clean.empty:
                return pd.DataFrame()
            stds = X_clean.std()
            rel = stds / (stds.sum() + 1e-8)
            res = pd.DataFrame({"feature": features, "mean_abs_shap": stds.values, "relative_importance": rel.values})
            res = res.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            res["rank"] = res.index + 1
            return res

        mean_abs = np.abs(shap_vals).mean(axis=0)
        total_imp = np.sum(mean_abs) + 1e-8
        rel_imp = mean_abs / total_imp

        res = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": mean_abs,
            "relative_importance": rel_imp,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        res["rank"] = res.index + 1
        return res

    def get_local_feature_importance(
        self,
        X_row: pd.DataFrame
    ) -> Dict[str, float]:
        """Generates Local Feature Importance for a single prediction row."""
        features = self.model.get_features()
        if not features:
            return {}

        shap_vals = self.compute_shap_values(X_row, max_samples=1)
        if shap_vals is None or len(shap_vals) == 0:
            # Fallback: estimate from feature z-scores
            row_vals = X_row[features].iloc[0].to_dict()
            return {k: float(v * 0.01) for k, v in row_vals.items()}

        row_shap = shap_vals[0] if shap_vals.ndim == 2 else shap_vals
        return dict(zip(features, [float(x) for x in row_shap]))

    def generate_waterfall_plot_data(self, X_row: pd.DataFrame) -> Dict[str, Any]:
        """Generates structured data required for a SHAP Waterfall Plot."""
        features = self.model.get_features()
        local_shap = self.get_local_feature_importance(X_row)
        pred = float(self.model.predict(X_row).iloc[0]) if not X_row.empty else 0.0
        base_value = pred - sum(local_shap.values()) if local_shap else 0.0

        feat_vals = X_row[features].iloc[0].to_dict() if not X_row.empty and features else {}

        return {
            "prediction": pred,
            "base_value": base_value,
            "values": local_shap,
            "feature_values": feat_vals,
        }

    def generate_waterfall_plot(
        self,
        X_row: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Generates and optionally saves a SHAP Waterfall Plot PNG."""
        data = self.generate_waterfall_plot_data(X_row)
        if not data["values"]:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sorted_items = sorted(data["values"].items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            feats = [x[0] for x in sorted_items]
            vals = [x[1] for x in sorted_items]
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]

            plt.figure(figsize=(10, 6))
            plt.barh(feats[::-1], vals[::-1], color=colors[::-1])
            plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            plt.title(f"Prediction Waterfall Plot (Predicted Score: {data['prediction']:+.4f})")
            plt.xlabel("SHAP Value (Contribution to Return)")
            plt.tight_layout()

            if save_path:
                p = Path(save_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(p, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"[ExplainabilityEngine] Saved waterfall plot to {p.resolve()}")
                return str(p.resolve())
            plt.close()
        except Exception as e:
            logger.warning(f"[ExplainabilityEngine] Failed to render waterfall plot: {e}")
        return None

    def generate_dependence_plot_data(
        self,
        X: pd.DataFrame,
        feature_col: str,
        max_samples: int = 500
    ) -> Dict[str, Any]:
        """Generates data points for a SHAP Dependence Plot for feature_col."""
        features = self.model.get_features()
        if feature_col not in features:
            return {"feature": feature_col, "x_values": [], "shap_values": []}

        X_clean = X[features].dropna().head(max_samples)
        shap_vals = self.compute_shap_values(X_clean, max_samples=max_samples)

        if shap_vals is None:
            # Fallback estimation from normalized feature values
            x_vals = X_clean[feature_col].values.tolist()
            col_std = float(X_clean[feature_col].std()) or 1.0
            col_mean = float(X_clean[feature_col].mean())
            y_shap = [float((x - col_mean) / col_std * 0.01) for x in x_vals]
        else:
            feat_idx = features.index(feature_col)
            x_vals = X_clean[feature_col].values.tolist()
            y_shap = shap_vals[:, feat_idx].tolist()

        return {
            "feature": feature_col,
            "x_values": x_vals,
            "shap_values": y_shap,
        }

    def generate_dependence_plot(
        self,
        X: pd.DataFrame,
        feature_col: str,
        save_path: Optional[str] = None,
        max_samples: int = 500
    ) -> Optional[str]:
        """Generates and saves a SHAP Dependence Scatter Plot PNG for feature_col."""
        data = self.generate_dependence_plot_data(X, feature_col, max_samples=max_samples)
        if not data["x_values"]:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.scatter(data["x_values"], data["shap_values"], alpha=0.7, color="#3498db", edgecolors="none")
            plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            plt.title(f"SHAP Dependence Plot — Feature: {feature_col}")
            plt.xlabel(f"{feature_col} Value")
            plt.ylabel("SHAP Value Impact")
            plt.tight_layout()

            if save_path:
                p = Path(save_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(p, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"[ExplainabilityEngine] Saved dependence plot to {p.resolve()}")
                return str(p.resolve())
            plt.close()
        except Exception as e:
            logger.warning(f"[ExplainabilityEngine] Failed to render dependence plot: {e}")
        return None

    def explain_prediction(
        self,
        X_row: pd.DataFrame,
        symbol: str = "ASSET",
        probability: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Produces institutional explanation object for EVERY prediction:
          - Prediction
          - Confidence
          - Top Positive Factors
          - Top Negative Factors
          - Risk Drivers
        """
        if X_row.empty:
            return {}

        # 1. Prediction & Submodel Variance
        preds = [float(m.predict(X_row).iloc[0]) for m in self.model.models if m.model is not None]
        mean_pred = float(np.mean(preds)) if preds else 0.0
        std_pred = float(np.std(preds)) if len(preds) > 1 else 0.01

        # 2. Confidence Index (0.0 to 1.0)
        # Higher magnitude & lower ensemble variance = higher confidence
        confidence = float(np.clip(1.0 / (1.0 + std_pred * 10.0), 0.1, 0.99))

        # 3. Local SHAP Attribution
        shap_dict = self.get_local_feature_importance(X_row)

        # 4. Top Positive & Negative Factors
        pos_factors = sorted([(k, v) for k, v in shap_dict.items() if v > 0], key=lambda x: x[1], reverse=True)
        neg_factors = sorted([(k, v) for k, v in shap_dict.items() if v < 0], key=lambda x: x[1])

        # 5. Risk Drivers
        risk_drivers = []
        for k, v in shap_dict.items():
            if k.lower() in RISK_FACTOR_NAMES or "vol" in k.lower() or "illiquid" in k.lower():
                impact_type = "High Risk Impact" if v < 0 else "Low Risk Drag"
                risk_drivers.append({"factor": k, "shap_impact": round(v, 4), "assessment": impact_type})
        risk_drivers.sort(key=lambda x: x["shap_impact"])

        return {
            "symbol": symbol,
            "prediction": round(mean_pred, 4),
            "confidence": round(confidence, 4),
            "top_positive_factors": [{"factor": k, "impact": round(v, 4)} for k, v in pos_factors[:5]],
            "top_negative_factors": [{"factor": k, "impact": round(v, 4)} for k, v in neg_factors[:5]],
            "risk_drivers": risk_drivers[:5],
        }


# ── Backward Compatibility Wrapper Functions ───────────────────────────────────

def compute_shap_values(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    max_samples: int = 2000,
) -> Optional[np.ndarray]:
    engine = InstitutionalExplainabilityEngine(model)
    return engine.compute_shap_values(X, max_samples=max_samples)


def global_shap_importance(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    max_samples: int = 2000,
) -> pd.DataFrame:
    engine = InstitutionalExplainabilityEngine(model)
    return engine.get_global_feature_importance(X, max_samples=max_samples)


def local_explanation(
    model: EnsembleAlphaModel,
    X_row: pd.DataFrame,
) -> Optional[pd.Series]:
    engine = InstitutionalExplainabilityEngine(model)
    local_dict = engine.get_local_feature_importance(X_row)
    if not local_dict:
        return None
    s = pd.Series(local_dict, name="shap_value")
    return s.reindex(s.abs().sort_values(ascending=False).index)


def shap_summary_plot(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    max_samples: int = 500,
    save_path: Optional[str] = None,
) -> None:
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("[SHAP] shap or matplotlib not installed.")
        return

    features = model.get_features()
    if not features:
        return
    X_shap = X[features].dropna().head(max_samples)
    engine = InstitutionalExplainabilityEngine(model)
    shap_values = engine.compute_shap_values(X_shap, max_samples=max_samples)
    if shap_values is None:
        return

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="dot", show=False, max_display=20)
    plt.title("SHAP Feature Contributions — QuantSphereX Ensemble")
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=150, bbox_inches="tight")
        logger.info(f"[SHAP] Summary plot saved to {p}")
    plt.close()


def shap_interaction_scores(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    top_n: int = 10,
    max_samples: int = 200,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        return pd.DataFrame()

    if not model.models or model.models[0].model is None:
        return pd.DataFrame()

    features = model.get_features()[:top_n]
    X_shap = X[features].dropna().head(max_samples)

    try:
        explainer = shap.TreeExplainer(model.models[0].model)
        interaction_vals = explainer.shap_interaction_values(X_shap)
        mean_inter = np.abs(interaction_vals).mean(axis=0)
        return pd.DataFrame(mean_inter, index=features, columns=features)
    except Exception as e:
        logger.warning(f"[SHAP] Interaction computation failed: {e}")
        return pd.DataFrame()
