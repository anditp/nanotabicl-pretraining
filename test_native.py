"""Reference integrity and explicit accelerator selection, with no ML dependencies."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from reference import ROOT, verify_reference
from train import choose_device


class NativeTests(unittest.TestCase):
    def test_exact_original_experiment_is_preserved(self):
        identity = verify_reference()
        self.assertEqual(identity, json.loads((ROOT / "reference_contract.json").read_text())["experiment_identity"])

    def test_changed_prior_or_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            shutil.copy2(ROOT / "reference_contract.json", path)
            shutil.copytree(ROOT / "experiment", path / "experiment")
            prior = path / "experiment/pretraining/prior.py"
            prior.write_text(prior.read_text() + "\n# modified\n")
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                verify_reference(path)

    def test_auto_selects_cuda_then_mps_then_cpu(self):
        for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                                    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))
            self.assertEqual(choose_device("auto", torch), expected)

    def test_explicit_accelerator_never_falls_back_to_cpu(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        for device in ("cuda", "mps"):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                choose_device(device, torch)


if __name__ == "__main__":
    unittest.main()
