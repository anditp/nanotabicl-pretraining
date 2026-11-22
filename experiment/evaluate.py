"""External evaluator: identical context, preprocessing and splits for both models."""
import argparse
from pathlib import Path
import time
import numpy as np
import openml
import torch
import common
from model_runtime import load_wsp, load_baseline, packages, regression_scores
from source.interface import Regressor, target_stats
from report import write_report


def score_archive(path, train, test, y, expected_std):
    with np.load(path, allow_pickle=False) as saved:
        if (not np.array_equal(saved["train"], train) or not np.array_equal(saved["test"], test) or
            not np.array_equal(saved["target"], y[test]) or float(saved["train_std"]) != expected_std):
            raise RuntimeError("Saved predictions do not match frozen split/current data")
        return {name: regression_scores(saved["target"], saved[name], expected_std) for name in ("wsp", "pretrained")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    cfg = common.config()["evaluation"]
    manifest = common.require_prepared()
    inputs = common.read_json(args.inputs / "inputs.json")
    if inputs["identity"] != common.identity() or inputs["manifest_sha256"] != common.digest(common.FROZEN / "manifest.json"):
        raise RuntimeError("Evaluation inputs differ from prepared experiment")
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    runtime = {"identity": common.identity(), "inputs": inputs, "packages": packages(), "policy": cfg}
    runtime_path = directory / "runtime.json"
    if runtime_path.exists() and common.read_json(runtime_path) != runtime:
        raise RuntimeError("Evaluation runtime/packages/config changed; use a new evaluation directory")
    common.write_json(runtime_path, runtime)
    # Invalidate any old final aggregate before attempting a resume.
    common.write_json(directory / "summary.json", {"complete": False, "reason": "Evaluation running; no aggregate yet"})
    if (directory / "scores.csv").exists():
        (directory / "scores.csv").replace(directory / "scores.previous.csv")
    torch.set_num_threads(4)
    openml.config.set_root_cache_directory(str(common.ROOT / "cache/openml"))
    models = {"wsp": load_wsp(args.inputs / "weights", inputs["wsp"]["files"]),
              "pretrained": load_baseline(args.inputs / "baseline.pt", inputs["baseline"]["sha256"])}
    parameter_statistics = {}
    for name, model in models.items():
        count = sum(p.numel() for p in model.parameters())
        nonzero = sum(int(torch.count_nonzero(p)) for p in model.parameters())
        parameter_statistics[name] = {"parameters": count, "nonzero_parameters": nonzero, "zero_fraction": 1 - nonzero / count}
    common.write_json(directory / "parameter_statistics.json", parameter_statistics)
    folds, failures = [], []
    for task in manifest["tasks"]:
        print(f"Evaluating {task['dataset']} (task {task['task_id']})", flush=True)
        try:
            dataset = common.retry_download(lambda: openml.datasets.get_dataset(task["dataset_id"], download_data=True),
                                             f"dataset {task['dataset_id']}", cfg["openml_attempts"])
            if int(dataset.version) != task["version"] or dataset.md5_checksum != task["md5_checksum"]:
                raise RuntimeError("Dataset version differs from frozen metadata")
            X, y, _, _ = common.retry_download(
                lambda: dataset.get_data(dataset_format="dataframe", target=task["target"]),
                f"data {task['dataset_id']}", cfg["openml_attempts"])
            X = X.loc[:, task["predictors"]]
            y = np.asarray(y, dtype=np.float64)
            if len(X) != task["rows"] or y.shape != (len(X),) or not np.isfinite(y).all():
                raise ValueError("Dataset shape/target differs from frozen scalar regression definition")
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
                completion = path.with_suffix(".json")
                path.parent.mkdir(exist_ok=True)
                if path.exists() and completion.exists():
                    previous = common.read_json(completion)
                    if previous["prediction_sha256"] != common.digest(path) or previous["split_sha256"] != split["sha256"]:
                        raise RuntimeError("Saved prediction checksum differs; refuse reuse")
                    metrics = score_archive(path, train, test, y, std)
                    seconds = previous["inference_seconds"]
                    print("  Reusing", key, flush=True)
                else:
                    predictions, seconds = {}, {}
                    for name, model in models.items():
                        start = time.monotonic()
                        regressor = Regressor(model, device=cfg["device"], test_batch_size=cfg["test_batch_size"])
                        regressor.fit(X.iloc[train], y[train])
                        predictions[name] = regressor.predict(X.iloc[test])
                        seconds[name] = time.monotonic() - start
                    temporary = path.with_suffix(".pending")
                    with temporary.open("wb") as f:
                        np.savez_compressed(f, train=train, test=test, target=y[test], train_std=std, **predictions)
                    temporary.replace(path)
                    metrics = score_archive(path, train, test, y, std)
                    common.write_json(completion, {"prediction_sha256": common.digest(path), "split_sha256": split["sha256"],
                                                  "metrics": metrics, "inference_seconds": seconds})
                    print("  Finished", key, metrics, flush=True)
                folds.append({"task_id": task["task_id"], "repeat": split["repeat"], "fold": split["fold"], "inference_seconds": seconds, **metrics})
                common.write_json(directory / "folds.json", folds)
            except Exception as error:
                failures.append({"task_id": task["task_id"], "repeat": split["repeat"], "fold": split["fold"], "error": repr(error)})
                common.write_json(directory / "failures.json", failures)
                print("  FAILED:", error, flush=True)
    common.write_json(directory / "failures.json", failures)
    if failures:
        common.write_json(directory / "summary.json", {"complete": False, "failures": failures,
                          "completed_splits": len(folds), "reason": "Evaluation incomplete; no aggregate reported"})
        raise RuntimeError("Some tasks/folds failed. Resume the same evaluation to reuse completed predictions.")
    summary = write_report(directory, manifest, folds)
    summary["wsp_audit_status"] = inputs["wsp"]["audit_status"]
    summary["parameter_statistics"] = parameter_statistics
    summary["inference_seconds"] = {name: sum(f["inference_seconds"][name] for f in folds) for name in models}
    summary["baseline_description"] = "Locally pretrained original nanoTabICL backbone, scalar MSE head; not an official pretrained checkpoint"
    common.write_json(directory / "summary.json", summary)
    print("Complete:", directory / "scores.csv", flush=True)


if __name__ == "__main__":
    main()
