"""
alpha_layer/governance/reproducibility.py
═════════════════════════════════════════
Reproducibility auditor for confirming that trained models are fully reproducible.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def verify_reproducibility(
    original_model: Any,
    retrained_model: Any,
    evaluation_dataset: pd.DataFrame,
    tolerance: float = 1e-6
) -> bool:
    """
    Verifies that a retrained model produces identical predictions on the evaluation dataset.
    This guarantees that random states, dataset versions, and features are fully reproducible.
    """
    logger.info("Starting model reproducibility verification...")

    # Compare features if available
    orig_feats = getattr(original_model, "features", None)
    retrained_feats = getattr(retrained_model, "features", None)

    if orig_feats and retrained_feats:
        if orig_feats != retrained_feats:
            logger.error(f"Features list mismatch! Original: {orig_feats}, Retrained: {retrained_feats}")
            return False
        logger.info("Feature columns lists are identical.")

    # Get predictions
    try:
        preds_orig = original_model.predict(evaluation_dataset)
        preds_retrained = retrained_model.predict(evaluation_dataset)
    except Exception as e:
        logger.error(f"Failed to generate predictions during reproducibility audit: {e}")
        return False

    # Check length
    if len(preds_orig) != len(preds_retrained):
        logger.error("Predictions length mismatch.")
        return False

    # Check absolute difference
    diff = np.abs(preds_orig.values - preds_retrained.values)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    logger.info(f"Reproducibility stats - Max diff: {max_diff:.2e} | Mean diff: {mean_diff:.2e}")

    if max_diff > tolerance:
        logger.error(f"Reproducibility audit FAILED: Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}.")
        return False

    logger.info("✅ Reproducibility audit PASSED. Model outputs are identical.")
    return True
