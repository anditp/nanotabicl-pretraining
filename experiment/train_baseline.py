"""Independent synthetic-data baseline. Never mounted into WSP construction."""
import argparse
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import common
from model_runtime import make_model, packages
from pretraining.prior import rand_cat_sizes, rand_dataset_filtered
from source.interface import context_tensors


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def learning_rate(step, cfg):
    if step <= cfg["warmup_steps"]:
        return cfg["learning_rate"] * step / cfg["warmup_steps"]
    fraction = (step - cfg["warmup_steps"]) / (cfg["steps"] - cfg["warmup_steps"])
    return cfg["learning_rate"] * 0.5 * (1 + math.cos(math.pi * fraction))


def sample_table(cfg, device):
    def sample(low, high):
        return min(high, max(low, int(np.exp(np.random.uniform(np.log(low), np.log(high + 1))))))
    n = sample(cfg["min_context_rows"], cfg["max_context_rows"])
    d = sample(cfg["min_features"], cfg["max_features"])
    cat_sizes = rand_cat_sizes(d)
    table = rand_dataset_filtered(cat_sizes, [0], n + cfg["query_rows"])
    # Preserve numeric feature order (x_2 before x_10). Categorical indicators
    # from the prior are passed as strings so the shared categorical encoder runs.
    import pandas as pd
    X = pd.DataFrame({j: table[f"x_{j}"].reshape(-1).cpu().numpy() for j in range(d)})
    for j, size in enumerate(cat_sizes):
        if size:
            X[j] = X[j].map(lambda value: "category:" + str(value))
    y = table["y_0"].reshape(-1).cpu().numpy().astype(np.float64)
    x, context_y, (mean, scale, _) = context_tensors(X.iloc[:n], y[:n], X.iloc[n:], device)
    query_y = torch.as_tensor((y[n:] - mean) / scale, dtype=torch.float32, device=device)
    return x, context_y, query_y


def update(model, optimizer, cfg, device, step):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for _ in range(cfg["tables_per_step"]):
        x, y, target = sample_table(cfg, device)
        prediction = model(x, y).reshape(-1)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; checkpoint unchanged")
        (loss / cfg["tables_per_step"]).backward()
        losses.append(float(loss.detach().cpu()))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
    lr = learning_rate(step, cfg)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Non-finite parameter; checkpoint unchanged")
    return {"step": step, "loss": float(np.mean(losses)), "learning_rate": lr, "gradient_norm": float(norm)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    args = parser.parse_args()
    common.require_prepared()
    cfg = common.config()
    training = cfg["training"]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(training["threads"])
    random.seed(training["seed"])
    np.random.seed(training["seed"])
    torch.manual_seed(training["seed"])
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    directory = args.run_dir.resolve()
    metadata = {"kind": "baseline", "experiment_identity": common.identity(), "model_config": cfg["model"],
                "training": training, "device": device, "packages": packages(), "step": 0}
    if args.resume:
        # This is our own full training checkpoint, containing NumPy RNG objects.
        checkpoint = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
        for key in ("kind", "experiment_identity", "model_config", "training", "device", "packages"):
            if checkpoint[key] != metadata[key]:
                raise RuntimeError("Resume mismatch: " + key)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng(checkpoint["rng"])
        metadata["step"] = checkpoint["step"]
    else:
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError("New baseline directory must be empty; use --resume for checkpoints")
        directory.mkdir(parents=True, exist_ok=True)

    def save():
        atomic_checkpoint(directory / "latest.pt", {**metadata, "model_state": model.state_dict(),
                          "optimizer_state": optimizer.state_dict(), "rng": rng_state()})
        common.write_json(directory / "state.json", {**metadata, "phase": "pretraining_checkpoint"})

    if not args.resume:
        save()
    print(f"Baseline: {directory}; device={device}; completed step={metadata['step']}", flush=True)
    try:
        with (directory / "training.jsonl").open("a") as log:
            for step in range(metadata["step"] + 1, training["steps"] + 1):
                start = time.monotonic()
                result = update(model, optimizer, training, device, step)
                metadata["step"] = step
                result["seconds"] = time.monotonic() - start
                import json
                log.write(json.dumps(result) + "\n")
                log.flush()
                print(f"Step {step}/{training['steps']}: loss={result['loss']:.6g}, {result['seconds']:.1f}s", flush=True)
                if step % training["checkpoint_every"] == 0 or step == training["steps"]:
                    save()
        atomic_checkpoint(directory / "final.pt", {**metadata, "model_state": model.cpu().state_dict()})
        common.write_json(directory / "state.json", {**metadata, "phase": "pretraining_finished",
                          "checkpoint_sha256": common.digest(directory / "final.pt")})
        print("Pretraining finished:", directory / "final.pt", flush=True)
    except BaseException:
        # Do not save a partially applied optimizer update or half a logical batch.
        print("Stopped. Resume from latest.pt; completed work after that checkpoint is replayed.", flush=True)
        raise


if __name__ == "__main__":
    main()
