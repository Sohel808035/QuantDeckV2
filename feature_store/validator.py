"""
feature_store/validator.py
───────────────────────────
Feature Validation and Data Quality Audit Engine for QuantSphereX Feature Store.
Enforces schema compliance, null budgets, range checks, infinity checks, and cross-sectional coverage.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from feature_store.schema import FeatureSchema, ValidationReport

logger = logging.getLogger(__name__)


class FeatureStoreValidator:
    """Audits feature panels against registered FeatureSchema definitions."""

    def __init__(self, schema: FeatureSchema):
        self.schema = schema

    def validate(self, panel: pd.DataFrame, min_tickers_per_date: int = 5) -> ValidationReport:
        """
        Runs comprehensive data quality audit on a feature panel DataFrame.
        Returns a ValidationReport detailing any warnings or fatal violations.
        """
        errors: Dict[str, List[str]] = {}
        warnings: List[str] = []
        total_rows = len(panel)

        if panel.empty:
            return ValidationReport(
                is_valid=False,
                total_rows=0,
                feature_errors={"panel": ["DataFrame is empty."]},
                warnings=["Empty DataFrame provided for validation."]
            )

        # 1. Structure Audit
        if isinstance(panel.index, pd.MultiIndex):
            levels = [str(l) for l in panel.index.names]
            if "Date" not in levels and "date" not in levels:
                warnings.append("MultiIndex does not explicitly name 'Date' level.")
        else:
            if panel.index.name not in ["Date", "date"]:
                warnings.append("Index is not named 'Date'.")

        # 2. Per-Feature Audits
        for feat_name, feat_def in self.schema.features.items():
            feat_errs = []

            # Check presence
            if feat_name not in panel.columns:
                feat_errs.append(f"Missing column '{feat_name}' required by schema.")
                errors[feat_name] = feat_errs
                continue

            col_data = panel[feat_name]

            # Null count audit
            null_count = int(col_data.isna().sum())
            null_pct = null_count / total_rows if total_rows > 0 else 0.0
            if not feat_def.allow_null and null_count > 0:
                feat_errs.append(f"Contains {null_count} NaNs but allow_null=False.")
            elif null_pct > feat_def.max_null_pct:
                feat_errs.append(
                    f"NaN ratio {null_pct:.1%} exceeds maximum threshold {feat_def.max_null_pct:.1%}."
                )

            # Infinity audit
            valid_numeric = col_data.dropna()
            if not valid_numeric.empty and np.issubdtype(valid_numeric.dtype, np.number):
                inf_count = int(np.isinf(valid_numeric).sum())
                if inf_count > 0:
                    feat_errs.append(f"Contains {inf_count} infinite values.")

                # Range audit
                if feat_def.min_value is not None:
                    below_min = int((valid_numeric < feat_def.min_value).sum())
                    if below_min > 0:
                        warnings.append(
                            f"Feature '{feat_name}': {below_min} rows below min_value ({feat_def.min_value})."
                        )

                if feat_def.max_value is not None:
                    above_max = int((valid_numeric > feat_def.max_value).sum())
                    if above_max > 0:
                        warnings.append(
                            f"Feature '{feat_name}': {above_max} rows above max_value ({feat_def.max_value})."
                        )

            if feat_errs:
                errors[feat_name] = feat_errs

        # 3. Cross-sectional Coverage Audit
        if isinstance(panel.index, pd.MultiIndex) and "Date" in panel.index.names:
            ticker_counts = panel.groupby(level="Date").size()
            low_coverage_dates = ticker_counts[ticker_counts < min_tickers_per_date]
            if not low_coverage_dates.empty:
                warnings.append(
                    f"Low ticker coverage (<{min_tickers_per_date} tickers) on {len(low_coverage_dates)} dates."
                )

        is_valid = len(errors) == 0
        if is_valid:
            logger.info(f"[Validator] Audit PASSED for schema '{self.schema.schema_id}' ({total_rows:,} rows)")
        else:
            logger.warning(f"[Validator] Audit FAILED for schema '{self.schema.schema_id}' with {len(errors)} errors")

        return ValidationReport(
            is_valid=is_valid,
            total_rows=total_rows,
            feature_errors=errors,
            warnings=warnings
        )
