"""Evaluate a completed pretrained baseline on the frozen CTR23 subset."""
import argparse
import datetime
from pathlib import Path
import shutil
import sys
import time
import uuid
from reference import ROOT, sha256, verify_reference
from train import choose_device
from evaluation.reporting import aggregate, write_scores


def code_identity():
    return {name: sha256(ROOT / name) for name in ("evaluate.py", "evaluation/reporting.py",
            "evaluation/pyproject.toml", "evaluation/uv.lock", "reference.py", "train.py")}


def stage_checkpoint(directory, source, common):
    record = common.validate_baseline_run(source)
    inputs = directory / "inputs"
    inputs.mkdir(parents=True)
    target = inputs / "baseline.pt"
    shutil.copy2(Path(record["path"]) / "final.pt", target)
    if sha256(target) != record["sha256"]:
        raise RuntimeError("Checkpoint changed while copying")
    metadata = {"model": "pretrained", "reference_identity": common.identity(),
                "checkpoint": record, "manifest_sha256": sha256(common.FROZEN / "manifest.json")}
    native = Path(record["path"]) / "native_provenance.json"
    if native.exists():
        metadata["native_producer"] = common.read_json(native)
    common.write_json(inputs / "inputs.json", metadata)
    return metadata


def score_archive(path, train, test, target, std, np, regression_scores):
    with np.load(path, allow_pickle=False) as saved:
        if (not np.array_equal(saved["train"], train) or not np.array_equal(saved["test"], test) or
            not np.array_equal(saved["target"], target[test]) or float(saved["train_std"]) != std):
            raise RuntimeError("Saved predictions do not match split/data/context scale")
        return regression_scores(saved["target"], saved["prediction"], std)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, help="Completed native or original baseline run directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-evaluation", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), help="Default CPU for a new run; resume retains its device")
    args = parser.parse_args()
    if args.resume_evaluation and (args.baseline_run or args.output):
        parser.error("Resume uses its saved checkpoint snapshot; omit --baseline-run and --output")
    if not args.resume_evaluation and args.baseline_run is None:
        parser.error("Provide --baseline-run or --resume-evaluation")
    verify_reference()
    sys.path.insert(0, str(ROOT / "experiment"))
    import numpy as np
    import openml
    import torch
    import common
    from model_runtime import load_baseline, packages, regression_scores
    from source.interface import Regressor, target_stats
    manifest = common.require_prepared()
    if args.resume_evaluation:
        directory = args.resume_evaluation.resolve()
        inputs = common.read_json(directory / "inputs/inputs.json")
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        directory = (args.output or ROOT / "evaluations" / stamp).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        inputs = stage_checkpoint(directory, args.baseline_run, common)
    if (inputs["model"] != "pretrained" or inputs["reference_identity"] != common.identity() or
        inputs["manifest_sha256"] != sha256(common.FROZEN / "manifest.json")):
        raise RuntimeError("Checkpoint inputs or frozen benchmark identity changed")
    previous = common.read_json(directory / "runtime.json") if (directory / "runtime.json").exists() else None
    device = choose_device(args.device or (previous["device"] if previous else "cpu"), torch)
    runtime = {"reference_identity": common.identity(), "inputs": inputs, "code": code_identity(),
               "packages": packages(), "device": device, "dtype": "float32", "policy": {**common.config()["evaluation"], "device": device}}
    if previous is not None and previous != runtime:
        raise RuntimeError("Evaluation code, environment, device or inputs changed; use a new output directory")
    common.write_json(directory / "runtime.json", runtime)
    common.write_json(directory / "summary.json", {"complete": False, "reason": "Evaluation running; no aggregate yet"})
    if (directory / "scores.csv").exists():
        (directory / "scores.csv").replace(directory / "scores.previous.csv")
    torch.set_num_threads(4)
    cache = ROOT / "cache/openml"
    cache.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(cache))
    model = load_baseline(directory / "inputs/baseline.pt", inputs["checkpoint"]["sha256"]).to(device)
    count = sum(p.numel() for p in model.parameters())
    nonzero = sum(int(torch.count_nonzero(p)) for p in model.parameters())
    parameter_statistics = {"parameters": count, "nonzero_parameters": nonzero, "zero_fraction": 1 - nonzero / count}
    folds, failures = [], []
    print("Evaluation:", directory, "device:", device, flush=True)
    for task in manifest["tasks"]:
        print(f"Evaluating {task['dataset']} (task {task['task_id']})", flush=True)
        try:
            dataset = common.retry_download(lambda: openml.datasets.get_dataset(task["dataset_id"], download_data=True), "dataset metadata")
            if int(dataset.version) != task["version"] or dataset.md5_checksum != task["md5_checksum"]:
                raise RuntimeError("OpenML dataset version/checksum changed")
            X, y, _, _ = common.retry_download(lambda: dataset.get_data(dataset_format="dataframe", target=task["target"]), "dataset data")
            X, y = X.loc[:, task["predictors"]], np.asarray(y, dtype=np.float64)
            if len(X) != task["rows"] or y.shape != (len(X),) or not np.isfinite(y).all():
                raise ValueError("Dataset dimensions or finite scalar target differ from frozen task")
        except Exception as error:
            failures.append({"task_id": task["task_id"], "stage": "download", "error": repr(error)})
            common.write_json(directory / "failures.json", failures)
            print("  FAILED:", error, flush=True)
            continue
        for split in task["splits"]:
            key = f"{task['task_id']}-r{split['repeat']}-f{split['fold']}"
            try:
                with np.load(common.FROZEN / split["file"], allow_pickle=False) as indices:
                    train, test = indices["train"], indices["test"]
                std = target_stats(y[train])[2]
                path = directory / "predictions" / (key + ".npz")
                path.parent.mkdir(exist_ok=True)
                completion = path.with_suffix(".json")
                if path.exists() and completion.exists():
                    saved = common.read_json(completion)
                    if saved["prediction_sha256"] != sha256(path) or saved["split_sha256"] != split["sha256"]:
                        raise RuntimeError("Saved prediction/split checksum changed")
                    metrics = score_archive(path, train, test, y, std, np, regression_scores)
                    seconds = saved["inference_seconds"]
                    print("  Reusing", key, flush=True)
                else:
                    start = time.monotonic()
                    regressor = Regressor(model, device=device, test_batch_size=runtime["policy"]["test_batch_size"])
                    regressor.fit(X.iloc[train], y[train])
                    prediction = regressor.predict(X.iloc[test])
                    seconds = time.monotonic() - start
                    with path.with_suffix(".pending").open("wb") as f:
                        np.savez_compressed(f, train=train, test=test, target=y[test], train_std=std, prediction=prediction)
                    path.with_suffix(".pending").replace(path)
                    metrics = score_archive(path, train, test, y, std, np, regression_scores)
                    common.write_json(completion, {"prediction_sha256": sha256(path), "split_sha256": split["sha256"],
                                                  "metrics": metrics, "inference_seconds": seconds})
                    print("  Finished", key, metrics, flush=True)
                folds.append({"task_id": task["task_id"], "repeat": split["repeat"], "fold": split["fold"],
                              "pretrained": metrics, "inference_seconds": seconds})
                common.write_json(directory / "folds.json", folds)
            except Exception as error:
                failures.append({"task_id": task["task_id"], "repeat": split["repeat"], "fold": split["fold"], "error": repr(error)})
                common.write_json(directory / "failures.json", failures)
                print("  FAILED:", error, flush=True)
    common.write_json(directory / "failures.json", failures)
    if failures:
        common.write_json(directory / "summary.json", {"complete": False, "failures": failures, "completed_splits": len(folds)})
        raise RuntimeError("Evaluation incomplete; no aggregate reported. Resume the same output after resolving failures.")
    rows, summary = aggregate(manifest, folds, common.baseline_table())
    summary.update({"parameter_statistics": parameter_statistics, "device": device,
                    "inference_seconds": sum(f["inference_seconds"] for f in folds),
                    "baseline_description": "Locally pretrained original nanoTabICL backbone, scalar MSE head; not an official pretrained checkpoint"})
    write_scores(directory, rows)
    common.write_json(directory / "summary.json", summary)
    print("Complete:", directory / "scores.csv", flush=True)


if __name__ == "__main__":
    main()
