"""Check the exact frozen WSP experiment without accessing benchmark datasets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_reference(root=ROOT):
    root = Path(root)
    contract = json.loads((root / "reference_contract.json").read_text())
    expected = contract["experiment_identity"]
    hashes = {}
    for filename, sha in expected["files"].items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid reference path")
        actual = sha256(root / "experiment" / relative)
        if actual != sha:
            raise RuntimeError("Frozen experiment source changed: " + filename)
        hashes[filename] = actual
    identity = {"files": hashes, "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    if identity != expected:
        raise RuntimeError("Reference experiment identity differs")
    cfg = json.loads((root / "experiment/config.json").read_text())
    if cfg != contract["configuration"] or cfg["source"]["commit"] != contract["upstream_commit"]:
        raise RuntimeError("Reference configuration differs")
    return identity


if __name__ == "__main__":
    print(json.dumps(verify_reference(), indent=2))
