"""
tests/unit/test_feature_store.py
──────────────────────────────────
Unit Test Suite for Institutional Feature Store.
"""

import unittest
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from feature_store import (
    FeatureDefinition,
    FeatureSchema,
    FeatureVersion,
    FeatureRegistry,
    FeatureStoreValidator,
    LineageTracker,
    FeatureStoreCache,
    FeatureStoreWriter,
    FeatureStoreReader,
)


class TestFeatureStore(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.registry_path = self.test_dir / "registry.json"
        self.store_dir = self.test_dir / "store"
        self.lineage_path = self.test_dir / "lineage.jsonl"

        self.registry = FeatureRegistry(registry_path=self.registry_path, store_dir=self.store_dir)
        self.lineage_tracker = LineageTracker(lineage_path=self.lineage_path)
        self.cache = FeatureStoreCache(max_entries=3)

        self.writer = FeatureStoreWriter(
            registry=self.registry,
            lineage_tracker=self.lineage_tracker,
            store_dir=self.store_dir
        )
        self.reader = FeatureStoreReader(
            registry=self.registry,
            cache=self.cache,
            store_dir=self.store_dir
        )

        # Generate mock multi-index feature panel
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])

        self.mock_panel = pd.DataFrame(index=idx)
        from feature_layer.implementations import FEATURE_COLS
        for col in FEATURE_COLS:
            self.mock_panel[col] = np.random.uniform(0.0, 1.0, size=len(idx))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_schema_and_registry(self):
        schema = FeatureRegistry.create_default_schema(schema_id="test_schema", version="1.0.0")
        self.registry.register_schema(schema)

        fetched = self.registry.get_schema("test_schema", "1.0.0")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.schema_id, "test_schema")
        self.assertIn("return_1m", fetched.features)

    def test_validator_pass_and_fail(self):
        schema = FeatureRegistry.create_default_schema()
        validator = FeatureStoreValidator(schema)

        # Valid panel
        report_pass = validator.validate(self.mock_panel)
        self.assertTrue(report_pass.is_valid)

        # Panel with Inf -> should fail
        corrupted_panel = self.mock_panel.copy()
        corrupted_panel.loc[corrupted_panel.index[0], "return_1m"] = np.inf
        report_fail = validator.validate(corrupted_panel)
        self.assertFalse(report_fail.is_valid)
        self.assertIn("return_1m", report_fail.feature_errors)

    def test_lineage_tracker(self):
        lid = self.lineage_tracker.record_lineage(
            input_dataset_name="stock_panel",
            input_dataset_version="v1.0",
            output_feature_version="feat_v1",
            transform_name="FactorEngine"
        )
        self.assertIsNotNone(lid)

        upstream = self.lineage_tracker.get_upstream("feat_v1")
        self.assertEqual(len(upstream), 1)
        self.assertEqual(upstream[0]["input_dataset_name"], "stock_panel")

    def test_cache_lru_behavior(self):
        cache = FeatureStoreCache(max_entries=2)
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})
        df3 = pd.DataFrame({"a": [3]})

        cache.put("v1", ("T1",), "2024-01-01", "2024-01-02", df1)
        cache.put("v2", ("T1",), "2024-01-01", "2024-01-02", df2)

        # Hit v1
        h1 = cache.get("v1", ("T1",), "2024-01-01", "2024-01-02")
        self.assertIsNotNone(h1)

        # Put v3 -> should evict v2 (since v1 was refreshed by hit)
        cache.put("v3", ("T1",), "2024-01-01", "2024-01-02", df3)
        self.assertIsNone(cache.get("v2", ("T1",), "2024-01-01", "2024-01-02"))
        self.assertIsNotNone(cache.get("v1", ("T1",), "2024-01-01", "2024-01-02"))

    def test_writer_and_reader_roundtrip(self):
        # 1. Write feature panel
        manifest = self.writer.write(
            panel=self.mock_panel,
            version_id="v_test_100",
            input_dataset_name="raw_stocks",
            input_dataset_version="raw_v1"
        )
        self.assertEqual(manifest.version_id, "v_test_100")
        self.assertEqual(manifest.row_count, len(self.mock_panel))

        # 2. Read full training set
        loaded_df = self.reader.load_training_set(version_id="v_test_100")
        self.assertEqual(len(loaded_df), len(self.mock_panel))

        # 3. Point-in-Time Inference Slice
        target_date = "2024-01-02"
        cs_df = self.reader.load_inference_set(
            date=target_date,
            tickers=["RELIANCE.NS", "TCS.NS"],
            version_id="v_test_100"
        )
        self.assertEqual(len(cs_df), 2)
        self.assertIn("return_1m", cs_df.columns)
