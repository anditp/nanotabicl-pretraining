"""Unit-aware CTR23 aggregation, with no dependency on a model runtime."""
import csv
import math
from pathlib import Path
import statistics
import common


def aggregate(manifest, folds):
    expected = {(t["task_id"], s["repeat"], s["fold"]) for t in manifest["tasks"] for s in t["splits"]}
    keys = [(f["task_id"], f["repeat"], f["fold"]) for f in folds]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Evaluation incomplete; no aggregate can be reported")
    published = common.baseline_table()
    rows = []
    for task in manifest["tasks"]:
        row = {"task_id": task["task_id"], "dataset": task["dataset"], "splits": len(task["splits"])}
        task_folds = [f for f in folds if f["task_id"] == task["task_id"]]
        for model in ("wsp", "pretrained"):
            for metric in ("rmse", "mae", "r2", "normalized_rmse"):
                values = [f[model][metric] for f in task_folds if f[model][metric] is not None]
                row[f"{model}_{metric}_defined_splits"] = len(values)
                # Dataset scores require the metric to be defined on every fold.
                row[f"{model}_{metric}"] = statistics.mean(values) if len(values) == len(task_folds) else None
        for learner, score in published[task["published_baseline_key"]]["scores"].items():
            row[learner + "_rmse"] = score
        rows.append(row)
    summary = {"complete": True, "datasets": len(rows), "official_splits": len(folds),
               "aggregation": "Arithmetic mean of fold RMSE within each dataset; unweighted datasets",
               "published_baselines": "Authors' reported CTR23 results, not local reruns",
               "metrics": {}, "paired_rmse_ratios": {}}
    for model in ("wsp", "pretrained"):
        summary["metrics"][model] = {}
        for metric in ("r2", "normalized_rmse"):
            values = [r[f"{model}_{metric}"] for r in rows if r[f"{model}_{metric}"] is not None]
            summary["metrics"][model][metric] = {"mean": statistics.mean(values) if values else None,
                 "median": statistics.median(values) if values else None, "defined_datasets": len(values),
                 "undefined_datasets": len(rows) - len(values)}
    comparators = ["pretrained"] + sorted(published[manifest["tasks"][0]["published_baseline_key"]]["scores"])
    for model in ("wsp", "pretrained"):
        for baseline in comparators:
            if baseline == model:
                continue
            ratios = []
            for row in rows:
                a, b = row[model + "_rmse"], row[baseline + "_rmse"]
                ratio = a / b if a > 0 and b > 0 else None
                row[f"{model}_over_{baseline}_rmse"] = ratio
                if ratio is not None:
                    ratios.append(ratio)
            summary["paired_rmse_ratios"][f"{model}_over_{baseline}"] = {
                "geometric_mean": math.exp(statistics.mean(map(math.log, ratios))) if ratios else None,
                "defined_datasets": len(ratios), "undefined_nonpositive_rmse": len(rows) - len(ratios),
                "interpretation": "Below one favors numerator"}
    return rows, summary


def write_report(directory, manifest, folds):
    rows, summary = aggregate(manifest, folds)
    directory = Path(directory)
    temporary = directory / "scores.csv.pending"
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / "scores.csv")
    common.write_json(directory / "summary.json", summary)
    return summary
