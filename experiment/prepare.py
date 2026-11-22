"""Freeze CTR23 metadata and official split indices without inspecting labels."""
import csv
import json
import re
import numpy as np
import openml
import common
from model_runtime import make_model, packages


def main():
    cfg = common.config()
    common.FROZEN.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(common.ROOT / "cache/openml"))
    identity = common.identity()
    # Constructor and parameter metadata only: no forward pass.
    model = make_model()
    shapes = {name: list(p.shape) for name, p in model.named_parameters()}
    common.write_json(common.FROZEN / "parameters.json", {
        "model_config": cfg["model"], "shapes": shapes, "count": common.parameter_count(shapes),
        "constructor_verified": True, "source_commit": cfg["source"]["commit"]})
    del model
    with (common.ROOT / "benchmarks/ctr23_task_ids.csv").open(newline="") as f:
        candidates = [int(row["task_id"]) for row in csv.DictReader(f)]
    if len(candidates) != 35 or len(set(candidates)) != 35:
        raise ValueError("CTR23 source pool must have 35 unique tasks")
    published = common.baseline_table()
    progress_path = common.FROZEN / "progress.json"
    progress = common.read_json(progress_path) if progress_path.exists() else {}
    if progress.get("identity") != identity:
        progress = {"identity": identity, "tasks": {}}
    selected, excluded = [], []
    for task_id in candidates:
        previous = progress["tasks"].get(str(task_id))
        if previous and all((common.FROZEN / s["file"]).exists() and common.digest(common.FROZEN / s["file"]) == s["sha256"] for s in previous.get("splits", [])):
            (selected if previous["eligible"] else excluded).append(previous)
            print("Reusing frozen task metadata:", task_id, flush=True)
            continue
        print("Preparing OpenML task", task_id, flush=True)
        task = common.retry_download(lambda: openml.tasks.get_task(task_id, download_splits=False, download_data=False, download_qualities=True, download_features_meta_data=True), f"task {task_id}")
        if getattr(task.task_type_id, "value", task.task_type_id) != 2:
            raise ValueError("CTR23 task is not supervised regression")
        dataset = common.retry_download(lambda: task.get_dataset(download_data=False), f"dataset metadata {task_id}")
        target = task.target_name
        targets = {part.strip() for part in target.split(",")}
        if len(targets) != 1:
            raise ValueError("Scalar regression requires one target")
        ignored = dataset.ignore_attribute or []
        ignored = [ignored] if isinstance(ignored, str) else ignored
        row_ids = dataset.row_id_attribute or []
        row_ids = [row_ids] if isinstance(row_ids, str) else row_ids
        excluded_columns = targets | set(ignored) | set(row_ids)
        predictors = [feature.name for feature in dataset.features.values() if feature.name not in excluded_columns]
        n = int(dataset.qualities["NumberOfInstances"])
        d = len(predictors)
        key = re.sub(r"[^a-z0-9]", "", dataset.name.lower())
        record = {"task_id": task_id, "dataset_id": int(dataset.dataset_id), "version": int(dataset.version),
                  "md5_checksum": dataset.md5_checksum,
                  "dataset": dataset.name, "target": target, "predictors": predictors, "rows": n, "features": d,
                  "eligible": n <= cfg["evaluation"]["max_total_rows"] and d <= cfg["evaluation"]["max_predictors"]}
        if not record["eligible"]:
            record["reason"] = "exceeds predeclared row/feature limits"
            excluded.append(record)
        else:
            if key not in published:
                raise ValueError(f"No unambiguous published-baseline dataset match for {dataset.name}; resolve metadata mapping first")
            record["published_baseline_key"] = key
            repeats, folds, samples = common.retry_download(task.get_split_dimensions, f"split dimensions {task_id}")
            if samples != 1:
                raise ValueError("Unexpected learning-curve samples in CTR23 task")
            record["split_dimensions"] = {"repeats": repeats, "folds": folds, "samples": samples}
            record["splits"] = []
            for repeat in range(repeats):
                for fold in range(folds):
                    train, test = common.retry_download(
                        lambda: task.get_train_test_split_indices(repeat=repeat, fold=fold, sample=0),
                        f"indices {task_id}/{repeat}/{fold}")
                    train, test = np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)
                    if (len(train) + len(test) != n or len(np.intersect1d(train, test)) or
                        not np.array_equal(np.sort(np.concatenate((train, test))), np.arange(n))):
                        raise ValueError("Official split is not a disjoint partition of the dataset")
                    relative = f"splits/{task_id}-r{repeat}-f{fold}-s0.npz"
                    path = common.FROZEN / relative
                    path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(path, train=train, test=test)
                    record["splits"].append({"repeat": repeat, "fold": fold, "sample": 0, "file": relative,
                                               "sha256": common.digest(path), "train_rows": len(train), "test_rows": len(test)})
            selected.append(record)
        progress["tasks"][str(task_id)] = record
        common.write_json(progress_path, progress)
    manifest = {"complete": True, "identity": identity, "suite_id": 353,
                "metadata_packages": packages(), "tasks": selected, "excluded": excluded,
                "policy": cfg["evaluation"], "parameter_manifest_sha256": common.digest(common.FROZEN / "parameters.json")}
    common.write_json(common.FROZEN / "manifest.json", manifest)
    common.require_prepared()
    print(f"Frozen {len(selected)} regression datasets, {sum(len(t['splits']) for t in selected)} official splits; {len(excluded)} excluded")


if __name__ == "__main__":
    main()
