"""
ml_layer/experiment_tracker.py
────────────────────────────────
ML Pipeline: Advanced Experiment Tracking Module (Upgraded)

An institutional-grade, file-based experiment tracker that:
  - Logs training runs with metadata, hyperparameters, metrics (IC, Sharpe, Drawdown, Cost Drag), and feature importances.
  - Automatically records experiments to JSON Lines format.
  - Provides side-by-side comparison utilities.
  - Generates Markdown reports.
  - Integrates directly with the Model Registry.
"""

from __future__ import annotations
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """An institutional experiment run record."""
    run_id:             str
    experiment_name:    str
    status:             str               # 'running' | 'completed' | 'failed'
    train_ic:           float = 0.0
    val_ic:             float = 0.0
    overfit_score:      float = 0.0
    decile_sharpe:      float = 0.0
    ic_tstat:           float = 0.0
    n_train_rows:       int   = 0
    n_val_rows:         int   = 0
    elapsed_seconds:    float = 0.0
    params:             Dict[str, Any] = field(default_factory=dict)
    tags:               Dict[str, str] = field(default_factory=dict)
    notes:              str  = ""
    started_at:         str  = ""
    finished_at:        str  = ""
    
    # ── Upgraded Institutional Governance Fields ──
    dataset_version:    str = ""
    feature_version:    str = ""
    sharpe:             float = 0.0
    drawdown:           float = 0.0
    cost_drag:          float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_id:           Optional[str] = None
    model_version:      Optional[str] = None


class ExperimentTracker:
    """
    Upgraded Institutional Experiment Tracker.
    Persists run records as JSON Lines in:
        tracking_dir/{experiment_name}/runs.jsonl
    """

    def __init__(self, tracking_dir: str = "ml_layer/experiments"):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

    def _run_file(self, experiment_name: str) -> Path:
        exp_dir = self.tracking_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir / "runs.jsonl"

    # ── Run Lifecycle ──────────────────────────────────────────────────────────

    def start_run(
        self,
        experiment_name: str,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        notes: str = "",
    ) -> str:
        """Starts a new experiment run and returns the run_id."""
        run_id = str(uuid.uuid4())[:8]
        record = RunRecord(
            run_id=run_id,
            experiment_name=experiment_name,
            status="running",
            params=params or {},
            tags=tags or {},
            notes=notes,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._append_run(record)
        logger.info(f"[Tracker] Run started: {run_id} | experiment='{experiment_name}'")
        return run_id

    def log_metrics(
        self,
        run_id: str,
        experiment_name: str,
        metrics: Dict[str, Any],
        elapsed_seconds: float = 0.0,
    ) -> None:
        """Updates a run record with evaluation metrics and governance metadata."""
        record = self._load_run(run_id, experiment_name)
        if record is None:
            logger.warning(f"[Tracker] Run '{run_id}' not found.")
            return

        # Core Metrics
        record.train_ic      = float(metrics.get("train_ic", record.train_ic))
        record.val_ic        = float(metrics.get("val_ic", record.val_ic))
        record.overfit_score = float(metrics.get("overfit_score", record.overfit_score))
        record.decile_sharpe = float(metrics.get("decile_sharpe", record.decile_sharpe))
        record.ic_tstat      = float(metrics.get("ic_tstat", record.ic_tstat))
        record.n_train_rows  = int(metrics.get("n_train_rows", record.n_train_rows))
        record.n_val_rows    = int(metrics.get("n_val_rows", record.n_val_rows))
        
        # Upgraded Fields
        record.dataset_version    = str(metrics.get("dataset_version", record.dataset_version))
        record.feature_version    = str(metrics.get("feature_version", record.feature_version))
        record.sharpe             = float(metrics.get("sharpe", record.sharpe))
        record.drawdown           = float(metrics.get("drawdown", record.drawdown))
        record.cost_drag          = float(metrics.get("cost_drag", record.cost_drag))
        record.feature_importance = dict(metrics.get("feature_importance", record.feature_importance))
        record.model_id           = metrics.get("model_id", record.model_id)
        record.model_version      = metrics.get("model_version", record.model_version)

        record.elapsed_seconds = elapsed_seconds
        record.status        = "completed"
        record.finished_at   = time.strftime("%Y-%m-%dT%H:%M:%S")

        self._update_run(run_id, experiment_name, record)
        logger.info(
            f"[Tracker] Logged | run={run_id} | "
            f"Val IC={record.val_ic:.4f} | Net Sharpe={record.sharpe:.2f}"
        )

    def fail_run(self, run_id: str, experiment_name: str, error: str = "") -> None:
        """Marks a run as failed."""
        record = self._load_run(run_id, experiment_name)
        if record:
            record.status = "failed"
            record.notes += f" | ERROR: {error}"
            record.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._update_run(run_id, experiment_name, record)

    # ── Querying & Comparison ──────────────────────────────────────────────────

    def get_runs(self, experiment_name: str) -> pd.DataFrame:
        """Returns all runs for an experiment as a sorted DataFrame."""
        run_file = self._run_file(experiment_name)
        if not run_file.exists():
            return pd.DataFrame()

        rows = []
        with open(run_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        # Deduplicate: keep latest version of each run_id
        seen = {}
        for r in rows:
            seen[r["run_id"]] = r
        rows = list(seen.values())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        priority_cols = [
            "run_id", "status", "train_ic", "val_ic", "overfit_score",
            "decile_sharpe", "ic_tstat", "sharpe", "drawdown", "cost_drag",
            "elapsed_seconds", "started_at"
        ]
        cols = [c for c in priority_cols if c in df.columns]
        # Include remaining columns at the end
        remaining = [c for c in df.columns if c not in cols]
        return df[cols + remaining].sort_values("val_ic", ascending=False).reset_index(drop=True)

    def compare_runs(self, experiment_name: str, run_id_a: str, run_id_b: str) -> Dict[str, Any]:
        """Compare metrics, parameters, and feature importance of two runs."""
        run_a = self._load_run(run_id_a, experiment_name)
        run_b = self._load_run(run_id_b, experiment_name)

        if not run_a:
            raise ValueError(f"Run {run_id_a} not found in experiment {experiment_name}")
        if not run_b:
            raise ValueError(f"Run {run_id_b} not found in experiment {experiment_name}")

        # Metrics diff
        metrics_diff = {
            "val_ic_diff": run_b.val_ic - run_a.val_ic,
            "train_ic_diff": run_b.train_ic - run_a.train_ic,
            "sharpe_diff": run_b.sharpe - run_a.sharpe,
            "drawdown_diff": run_b.drawdown - run_a.drawdown,
            "cost_drag_diff": run_b.cost_drag - run_a.cost_drag,
        }

        # Parameters diff
        p_a = run_a.params or {}
        p_b = run_b.params or {}
        all_param_keys = set(p_a.keys()).union(p_b.keys())
        param_diff = {}
        for k in all_param_keys:
            val_a = p_a.get(k)
            val_b = p_b.get(k)
            if val_a != val_b:
                param_diff[k] = {"run_a": val_a, "run_b": val_b}

        # Feature Importance diff
        f_a = run_a.feature_importance or {}
        f_b = run_b.feature_importance or {}
        all_feat_keys = set(f_a.keys()).union(f_b.keys())
        feat_diff = {}
        for k in all_feat_keys:
            val_a = f_a.get(k, 0.0)
            val_b = f_b.get(k, 0.0)
            if val_a != val_b:
                feat_diff[k] = {"run_a": val_a, "run_b": val_b}

        return {
            "run_id_a": run_id_a,
            "run_id_b": run_id_b,
            "metrics_diff": metrics_diff,
            "param_diff": param_diff,
            "feat_diff": feat_diff
        }

    # ── Report Generation ──────────────────────────────────────────────────────

    def generate_report(self, experiment_name: str, run_id: str, output_path: Optional[str] = None) -> str:
        """Generates a Markdown experiment report and optionally writes it to output_path."""
        run = self._load_run(run_id, experiment_name)
        if not run:
            raise ValueError(f"Run {run_id} not found.")

        # Formatting feature importances
        feat_imp_md = "\n".join([f"- **{k}**: {v:.4f}" for k, v in sorted(run.feature_importance.items(), key=lambda x: x[1], reverse=True)])
        if not feat_imp_md:
            feat_imp_md = "No feature importance logged."

        md = f"""# QuantSphereX Experiment Report: Run {run_id}
**Experiment**: {experiment_name} | **Status**: {run.status}
**Started At**: {run.started_at} | **Finished At**: {run.finished_at}

## 📊 Summary Metrics
- **Validation IC**: {run.val_ic:.4f}
- **Train IC**: {run.train_ic:.4f}
- **Overfit Score**: {run.overfit_score:.4f}
- **Net Sharpe Ratio**: {run.sharpe:.2f}
- **Max Drawdown**: {run.drawdown:.2%}
- **Cost Drag**: {run.cost_drag:.1f} bps

## ⚙️ Hyperparameters
```json
{json.dumps(run.params, indent=2)}
```

## 🧠 Feature Importance
{feat_imp_md}

## 🛡️ Governance Metadata
- **Dataset Version**: {run.dataset_version or "N/A"}
- **Feature Version**: {run.feature_version or "N/A"}
- **Linked Model ID**: {run.model_id or "N/A"}
- **Linked Model Version**: {run.model_version or "N/A"}

## 📝 Notes
{run.notes or "No notes logged."}
"""
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(md, encoding="utf-8")
            logger.info(f"Experiment report saved to {p.resolve()}")

        return md

    # ── Model Registry Integration ─────────────────────────────────────────────

    def promote_to_registry(
        self,
        experiment_name: str,
        run_id: str,
        model_object: Any,
        model_registry: Any,
        model_id: str,
        version: str,
        status: str = "Production",
        notes: str = ""
    ) -> Any:
        """
        Promotes an experiment run to the Model Registry.
        Saves the model object and populates registry metadata directly from the run record.
        """
        run = self._load_run(run_id, experiment_name)
        if not run:
            raise ValueError(f"Run {run_id} not found in experiment {experiment_name}")

        from alpha_layer.governance.registry import save_model

        # Build registry metadata args from the run record
        metadata_args = {
            "model_id": model_id,
            "version": version,
            "dataset_version": run.dataset_version or "experiment_dataset",
            "feature_version": run.feature_version or "experiment_features",
            "hyperparameters": run.params,
            "validation_ic": run.val_ic,
            "train_ic": run.train_ic,
            "sharpe": run.sharpe,
            "drawdown": run.drawdown,
            "status": status,
            "notes": f"Promoted from experiment run {run_id}. {notes}"
        }

        # Save to registry
        saved_path = save_model(model_object, metadata_args, registry=model_registry)
        
        # Update experiment record with linked model details
        run.model_id = model_id
        run.model_version = version
        self._update_run(run_id, experiment_name, run)
        
        logger.info(f"[Tracker] Promoted run {run_id} to Model Registry at {saved_path}")
        return saved_path

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def export_csv(self, experiment_name: str, path: str) -> None:
        """Exports full run history to CSV."""
        df = self.get_runs(experiment_name)
        df.to_csv(path, index=False)
        logger.info(f"[Tracker] Exported {len(df)} runs to {path}")

    def _append_run(self, record: RunRecord) -> None:
        run_file = self._run_file(record.experiment_name)
        with open(run_file, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def _load_run(self, run_id: str, experiment_name: str) -> Optional[RunRecord]:
        run_file = self._run_file(experiment_name)
        if not run_file.exists():
            return None
        latest = None
        with open(run_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("run_id") == run_id:
                    latest = RunRecord(**data)
        return latest

    def _update_run(self, run_id: str, experiment_name: str, record: RunRecord) -> None:
        """Appends updated record (deduplication done at read time)."""
        self._append_run(record)
