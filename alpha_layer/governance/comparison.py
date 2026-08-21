"""
alpha_layer/governance/comparison.py
═════════════════════════════════════
Model comparison utilities for comparing two model versions side-by-side.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from alpha_layer.governance.registry import ModelMetadata, ModelRegistry

logger = logging.getLogger(__name__)

def compare_models(
    model_id: str,
    version_a: str,
    version_b: str,
    registry: Optional[ModelRegistry] = None
) -> Dict[str, Any]:
    """
    Compare performance, hyperparameters, and metadata of two models side-by-side.
    Returns a dictionary summarizing the comparison.
    """
    reg = registry or ModelRegistry()
    meta_a = reg.get_model_metadata(model_id, version_a)
    meta_b = reg.get_model_metadata(model_id, version_b)

    if not meta_a:
        raise ValueError(f"Model {model_id} version {version_a} not found.")
    if not meta_b:
        raise ValueError(f"Model {model_id} version {version_b} not found.")

    # Calculate differences for key metric values
    metrics_diff = {
        "train_ic_diff": meta_b.train_ic - meta_a.train_ic,
        "validation_ic_diff": meta_b.validation_ic - meta_a.validation_ic,
        "sharpe_diff": meta_b.sharpe - meta_a.sharpe,
        "drawdown_diff": meta_b.drawdown - meta_a.drawdown,
    }

    # Compare hyperparameters
    h_a = meta_a.hyperparameters
    h_b = meta_b.hyperparameters
    all_keys = set(h_a.keys()).union(h_b.keys())
    param_diff = {}
    for k in all_keys:
        val_a = h_a.get(k)
        val_b = h_b.get(k)
        if val_a != val_b:
            param_diff[k] = {"version_a": val_a, "version_b": val_b}

    report = {
        "model_id": model_id,
        "model_a": {
            "version": meta_a.version,
            "status": meta_a.status,
            "training_date": meta_a.training_date,
            "dataset_version": meta_a.dataset_version,
            "feature_version": meta_a.feature_version,
            "metrics": {
                "train_ic": meta_a.train_ic,
                "validation_ic": meta_a.validation_ic,
                "sharpe": meta_a.sharpe,
                "drawdown": meta_a.drawdown,
            }
        },
        "model_b": {
            "version": meta_b.version,
            "status": meta_b.status,
            "training_date": meta_b.training_date,
            "dataset_version": meta_b.dataset_version,
            "feature_version": meta_b.feature_version,
            "metrics": {
                "train_ic": meta_b.train_ic,
                "validation_ic": meta_b.validation_ic,
                "sharpe": meta_b.sharpe,
                "drawdown": meta_b.drawdown,
            }
        },
        "metrics_diff": metrics_diff,
        "hyperparameter_diff": param_diff,
    }

    # Log comparison report in a clean format
    logger.info("=" * 60)
    logger.info(f"MODEL COMPARISON REPORT: {model_id}")
    logger.info(f"Model A: Version {version_a} ({meta_a.status})")
    logger.info(f"Model B: Version {version_b} ({meta_b.status})")
    logger.info("-" * 60)
    logger.info(f"Metric          | {version_a:<10} | {version_b:<10} | Diff")
    logger.info("-" * 60)
    logger.info(f"Train IC        | {meta_a.train_ic:<10.4f} | {meta_b.train_ic:<10.4f} | {metrics_diff['train_ic_diff']:+.4f}")
    logger.info(f"Validation IC   | {meta_a.validation_ic:<10.4f} | {meta_b.validation_ic:<10.4f} | {metrics_diff['validation_ic_diff']:+.4f}")
    logger.info(f"Sharpe          | {meta_a.sharpe:<10.2f} | {meta_b.sharpe:<10.2f} | {metrics_diff['sharpe_diff']:+.2f}")
    logger.info(f"Drawdown        | {meta_a.drawdown:<10.2%} | {meta_b.drawdown:<10.2%} | {metrics_diff['drawdown_diff']:+.2%}")
    logger.info("=" * 60)

    return report
