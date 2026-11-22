"""Baseline-only scoring and offline evaluation/resume integration tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from reference import ROOT, verify_reference
from evaluation.reporting import aggregate


def load_evaluator():
    spec = importlib.util.spec_from_file_location("native_baseline_evaluator", ROOT / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReportTests(unittest.TestCase):
    def fixture(self):
        manifest = {"tasks": [{"task_id": 1, "dataset": "synthetic-fixture", "published_baseline_key": "fixture",
                   "splits": [{"repeat": 0, "fold": 0}, {"repeat": 0, "fold": 1}]}]}
        published = {"fixture": {"scores": {"ranger": 4, "rpart": 5, "xgboost": 3, "cv_glmnet": 6, "gam": 2}}}
        folds = [{"task_id": 1, "repeat": 0, "fold": i, "inference_seconds": .1,
                  "pretrained": {"rmse": rmse, "mae": 1, "r2": .5, "normalized_rmse": .4}}
                 for i, rmse in enumerate((1, 3))]
        return manifest, folds, published

    def test_pretrained_only_mean_fold_rmse_and_published_ratios(self):
        manifest, folds, published = self.fixture()
        rows, summary = aggregate(manifest, folds, published)
        self.assertEqual(rows[0]["pretrained_rmse"], 2)
        self.assertEqual(summary["models_evaluated"], ["pretrained"])
        self.assertEqual(summary["paired_rmse_ratios"]["pretrained_over_ranger"]["geometric_mean"], .5)
        self.assertFalse(any("wsp" in key for key in rows[0]))
        self.assertNotIn("mean_rmse", summary)

    def test_incomplete_or_duplicate_folds_never_aggregate(self):
        manifest, folds, published = self.fixture()
        for invalid in (folds[:-1], folds + [folds[0]], []):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                aggregate(manifest, invalid, published)

    def test_undefined_metrics_and_zero_ratios_are_counted(self):
        manifest, folds, published = self.fixture()
        folds[0]["pretrained"]["r2"] = None
        published["fixture"]["scores"]["ranger"] = 0
        rows, summary = aggregate(manifest, folds, published)
        self.assertIsNone(rows[0]["pretrained_r2"])
        self.assertEqual(summary["metrics"]["r2"]["undefined_datasets"], 1)
        self.assertEqual(summary["paired_rmse_ratios"]["pretrained_over_ranger"]["undefined_nonpositive_rmse"], 1)

    def test_frozen_benchmark_matches_original_subset(self):
        verify_reference()
        sys.path.insert(0, str(ROOT / "experiment"))
        import common
        manifest = common.require_prepared()
        self.assertEqual(len(manifest["tasks"]), 19)
        self.assertEqual(sum(len(t["splits"]) for t in manifest["tasks"]), 640)


@unittest.skipUnless(importlib.util.find_spec("openml") is not None, "Use the evaluation uv project for integration tests")
class EvaluationTests(unittest.TestCase):
    def setUp(self):
        verify_reference()
        sys.path.insert(0, str(ROOT / "experiment"))

    def test_unfinished_baseline_is_rejected(self):
        import common
        stage_checkpoint = load_evaluator().stage_checkpoint
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common.write_json(root / "baseline/state.json", {"kind": "baseline", "phase": "pretraining_checkpoint", "step": 100})
            with self.assertRaisesRegex(RuntimeError, "not finished"):
                stage_checkpoint(root / "evaluation", root / "baseline", common)

    def test_failed_fold_resumes_without_recomputing_completed_predictions(self):
        import numpy as np
        import pandas as pd
        import openml
        import torch
        sys.path.insert(0, str(ROOT / "experiment"))
        import common
        import model_runtime
        import source.interface as interface
        evaluate = load_evaluator()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "final.pt").write_bytes(b"test-only opaque checkpoint; loader mocked")
            common.write_json(baseline / "state.json", {"kind": "baseline", "phase": "pretraining_finished", "step": 20000,
                "experiment_identity": common.identity(), "checkpoint_sha256": common.digest(baseline / "final.pt")})
            frozen = root / "frozen"
            frozen.mkdir()
            splits = []
            for fold, (train, test) in enumerate((([0, 1], [2, 3]), ([2, 3], [0, 1]))):
                path = frozen / f"split-{fold}.npz"
                np.savez_compressed(path, train=np.asarray(train), test=np.asarray(test))
                splits.append({"repeat": 0, "fold": fold, "file": path.name, "sha256": common.digest(path)})
            manifest = {"tasks": [{"task_id": 1, "dataset_id": 1, "dataset": "synthetic-fixture", "version": 1,
                "md5_checksum": "fixture-checksum", "target": "y", "predictors": ["x"], "rows": 4,
                "published_baseline_key": "fixture", "splits": splits}]}
            common.write_json(frozen / "manifest.json", manifest)
            data = pd.DataFrame({"x": [1., 2., 3., 4.]})
            target = np.asarray([1., 2., 3., 4.])
            dataset = SimpleNamespace(version=1, md5_checksum="fixture-checksum", get_data=lambda **kw: (data, target, [], []))
            calls = []
            fail_second = [True]
            class Predictor:
                def __init__(self, *args, **kwargs):
                    pass
                def fit(self, X, y):
                    self.mean = float(np.mean(y))
                    return self
                def predict(self, X):
                    calls.append(self.mean)
                    if fail_second[0] and len(calls) == 2:
                        raise RuntimeError("test-only interrupted fold")
                    return np.full(len(X), self.mean)
            output = root / "evaluation"
            published = {"fixture": {"scores": {"ranger": 4, "rpart": 5, "xgboost": 3, "cv_glmnet": 6, "gam": 2}}}
            with patch.object(common, "FROZEN", frozen), patch.object(common, "require_prepared", return_value=manifest), \
                 patch.object(common, "baseline_table", return_value=published), patch.object(evaluate, "code_identity", return_value={"fixture": "v1"}), \
                 patch.object(openml.datasets, "get_dataset", return_value=dataset), patch.object(interface, "Regressor", Predictor), \
                 patch.object(model_runtime, "load_baseline", side_effect=lambda *args: torch.nn.Linear(1, 1)), patch("builtins.print"):
                with patch.object(sys, "argv", ["evaluate.py", "--baseline-run", str(baseline), "--output", str(output)]):
                    with self.assertRaisesRegex(RuntimeError, "incomplete"):
                        evaluate.main()
                self.assertFalse(common.read_json(output / "summary.json")["complete"])
                self.assertFalse((output / "scores.csv").exists())
                fail_second[0] = False
                calls.clear()
                with patch.object(sys, "argv", ["evaluate.py", "--resume-evaluation", str(output)]):
                    evaluate.main()
                self.assertEqual(len(calls), 1)
                summary = common.read_json(output / "summary.json")
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["models_evaluated"], ["pretrained"])
                self.assertNotIn("wsp", (output / "scores.csv").read_text())


if __name__ == "__main__":
    unittest.main()
