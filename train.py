"""Run the frozen original trainer natively, without CTR23 preparation or Docker."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from reference import ROOT, sha256, verify_reference


def choose_device(requested, torch):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check your NVIDIA driver and PyTorch build; no CPU fallback was used.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable. Run native ARM Python on an MPS-capable Mac; no CPU fallback was used.")
    return requested


def producer_metadata(device, torch):
    files = {name: sha256(ROOT / name) for name in ("train.py", "reference.py", "reference_contract.json", "pyproject.toml", "uv.lock")}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"entrypoint": "native_uv", "files": files,
            "reference_experiment_identity": verify_reference(),
            "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "platform": platform.platform(), "python": sys.version,
            "device": device,
            "accelerator": torch.cuda.get_device_name() if device == "cuda" else "Apple MPS" if device == "mps" else "CPU"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    identity = verify_reference()
    sys.path.insert(0, str(ROOT / "experiment"))
    import torch
    import common
    import train_baseline as trainer
    if common.identity() != identity:
        raise RuntimeError("Original trainer did not load the expected experiment")
    device = choose_device(args.device, torch)
    producer = producer_metadata(device, torch)
    if args.resume:
        checkpoint = torch.load(args.run_dir / "latest.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("native_producer", {}).get("files") != producer["files"]:
            raise RuntimeError("Native trainer changed, or checkpoint is from another environment. Resume the same native run; CPU-to-GPU migration is not an exact resume.")
        del checkpoint

    # Benchmark preparation was a coordination gate in the original project,
    # not a training input. Replace it with exact source/configuration checks.
    # No other part of the original training recipe or implementation changes.
    common.require_prepared = verify_reference
    original_save = trainer.atomic_checkpoint
    def save(path, checkpoint):
        original_save(path, {**checkpoint, "native_producer": producer})
        common.write_json(Path(path).parent / "native_provenance.json", producer)
    trainer.atomic_checkpoint = save

    # Instrument the unchanged sampler/update so the log distinguishes CPU
    # prior/filtering/transfer time from network optimization time.
    original_sample, original_update = trainer.sample_table, trainer.update
    sampling_seconds = [0.0]
    def sample(cfg, selected_device):
        start = time.monotonic()
        result = original_sample(cfg, selected_device)
        sampling_seconds[0] += time.monotonic() - start
        return result
    def update(*arguments):
        sampling_seconds[0] = 0.0
        start = time.monotonic()
        result = original_update(*arguments)
        result["prior_and_transfer_seconds"] = sampling_seconds[0]
        result["optimization_seconds"] = max(0., time.monotonic() - start - sampling_seconds[0])
        return result
    trainer.sample_table, trainer.update = sample, update
    print("Native device:", device, producer["accelerator"], flush=True)
    print("Reference experiment:", identity["sha256"], flush=True)
    sys.argv = ["train_baseline.py", "--run-dir", str(args.run_dir), "--device", device] + (["--resume"] if args.resume else [])
    trainer.main()


if __name__ == "__main__":
    main()
