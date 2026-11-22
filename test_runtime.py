"""Numerical integration of the native adapter, original trainer and checkpoints."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from reference import ROOT, verify_reference

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Run through uv to install the numerical dependencies")
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        verify_reference()
        sys.path.insert(0, str(ROOT / "experiment"))
        import common
        import train_baseline
        self.common, self.trainer = common, train_baseline
        torch.set_num_threads(2)

    def test_native_adapter_uses_source_checks_and_preserves_checkpoint_metadata(self):
        import train
        captured = {}
        def original_main():
            captured["source_check"] = self.common.require_prepared()
            captured["argv"] = sys.argv.copy()
            self.trainer.atomic_checkpoint(output / "final.pt", {"kind": "baseline", "step": 20000,
                                               "experiment_identity": self.common.identity(), "model_state": {"weight": torch.ones(2)}})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            save = self.trainer.atomic_checkpoint
            sample, update = self.trainer.sample_table, self.trainer.update
            gate = self.common.require_prepared
            try:
                with patch.object(sys, "argv", ["train.py", "--run-dir", str(output), "--device", "cpu"]), \
                     patch.object(self.trainer, "main", side_effect=original_main):
                    train.main()
            finally:
                self.trainer.atomic_checkpoint = save
                self.trainer.sample_table, self.trainer.update = sample, update
                self.common.require_prepared = gate
            saved = torch.load(output / "final.pt", weights_only=True)
            self.assertEqual(saved["experiment_identity"], verify_reference())
            self.assertEqual(saved["native_producer"]["entrypoint"], "native_uv")
            self.assertEqual(captured["source_check"], verify_reference())
            self.assertEqual(captured["argv"][-2:], ["--device", "cpu"])
            self.assertTrue((output / "native_provenance.json").is_file())

    def test_original_optimizer_and_rng_resume_exactly_on_cpu(self):
        from model_runtime import make_model
        torch.manual_seed(9)
        cfg = {**self.common.config()["model"], "embed_dim": 16, "col_num_blocks": 1, "row_num_blocks": 1,
               "icl_num_blocks": 1, "col_nhead": 2, "row_nhead": 2, "icl_nhead": 2, "n_cls_cols": 2, "n_cls_rows": 8}
        model = make_model(model_config=cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        def step():
            optimizer.zero_grad()
            prediction = model(torch.randn(1, 10, 3), torch.randn(1, 8)).reshape(-1)
            loss = torch.nn.functional.mse_loss(prediction, torch.randn(2))
            loss.backward()
            optimizer.step()
            return copy.deepcopy(model.state_dict())
        step()
        saved = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": self.trainer.rng_state()}
        expected = step()
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        self.trainer.restore_rng(saved["rng"])
        actual = step()
        self.assertTrue(all(torch.equal(actual[name], expected[name]) for name in actual))


if __name__ == "__main__":
    unittest.main()
