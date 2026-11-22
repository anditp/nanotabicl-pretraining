"""Matched CTR23 aggregation for the pretrained baseline and published learners."""
import csv
import math
from pathlib import Path
import statistics


def aggregate(manifest, folds, published):
    expected = {(t["task_id"], s["repeat"], s["fold"]) for t in manifest["tasks"] for s in t["splits"]}
    keys = [(f["task_id"], f["repeat"], f["fold"]) for f in folds]
    if not expected or len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Evaluation incomplete; no aggregate can be reported")
    rows = []
    for task in manifest["tasks"]:
        row = {"task_id": task["task_id"], "dataset": task["dataset"], "splits": len(task["splits"])}
        selected = [f for f in folds if f["task_id"] == task["task_id"]]
        for metric in ("rmse", "mae", "r2", "normalized_rmse"):
            values = [f["pretrained"][metric] for f in selected if f["pretrained"][metric] is not None]
            row[f"pretrained_{metric}_defined_splits"] = len(values)
            row[f"pretrained_{metric}"] = statistics.mean(values) if len(values) == len(selected) else None
        row["inference_seconds"] = sum(f["inference_seconds"] for f in selected)
        for learner, score in published[task["published_baseline_key"]]["scores"].items():
            row[learner + "_rmse"] = score
        rows.append(row)
    summary = {"complete": True, "models_evaluated": ["pretrained"], "datasets": len(rows),
               "official_splits": len(folds), "aggregation": "Arithmetic mean of fold RMSE per dataset; unweighted datasets",
               "published_baselines": "Authors' CTR23 scores, not local reruns", "metrics": {}, "paired_rmse_ratios": {}}
    for metric in ("r2", "normalized_rmse"):
        values = [r[f"pretrained_{metric}"] for r in rows if r[f"pretrained_{metric}"] is not None]
        summary["metrics"][metric] = {"mean": statistics.mean(values) if values else None,
             "median": statistics.median(values) if values else None, "defined_datasets": len(values),
             "undefined_datasets": len(rows) - len(values)}
    learners = sorted(published[manifest["tasks"][0]["published_baseline_key"]]["scores"])
    for learner in learners:
        ratios = []
        for row in rows:
            a, b = row["pretrained_rmse"], row[learner + "_rmse"]
            ratio = a / b if a > 0 and b > 0 else None
            row[f"pretrained_over_{learner}_rmse"] = ratio
            if ratio is not None:
                ratios.append(ratio)
        summary["paired_rmse_ratios"]["pretrained_over_" + learner] = {
            "geometric_mean": math.exp(statistics.mean(map(math.log, ratios))) if ratios else None,
            "defined_datasets": len(ratios), "undefined_nonpositive_rmse": len(rows) - len(ratios),
            "interpretation": "Below one favors the pretrained model"}
    return rows, summary


def write_scores(directory, rows):
    path = Path(directory) / "scores.csv"
    pending = path.with_suffix(".csv.pending")
    with pending.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)
