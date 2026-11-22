"""Disposable full-backbone training check on the requested native device."""
import argparse
import json
import sys
import time
from reference import ROOT, verify_reference
from train import choose_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    identity = verify_reference()
    sys.path.insert(0, str(ROOT / "experiment"))
    import torch
    import common
    from model_runtime import make_model, packages
    from train_baseline import update
    device = choose_device(args.device, torch)
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    torch.set_num_threads(common.config()["training"]["threads"])
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    # The preflight uses the full original architecture but a disposable small
    # table. The real fixed training budget/distribution remain unchanged.
    cfg = {**common.config()["training"], "tables_per_step": 1, "min_context_rows": 32,
           "max_context_rows": 32, "query_rows": 8, "min_features": 3, "max_features": 3}
    start = time.monotonic()
    result = update(model, optimizer, cfg, device, 1)
    result.update({"device": device, "seconds": time.monotonic() - start,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "packages": packages(), "reference_identity": identity["sha256"], "disposable_preflight": True})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
