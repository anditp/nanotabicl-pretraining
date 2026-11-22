"""Standard-library experiment configuration, integrity and protocol helpers."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "benchmarks/frozen"
CONSTRUCTION_FILES = ("model.py", "interface.py", "model_config.json", "LICENSE", "PROVENANCE.md")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def config():
    value = read_json(ROOT / "config.json")
    if value["model"]["max_classes"] != 0 or value["model"]["out_dim"] != 1:
        raise ValueError("This experiment is scalar regression: max_classes=0, out_dim=1")
    if value["model"] != read_json(ROOT / "source/model_config.json"):
        raise ValueError("config.json and source/model_config.json differ")
    supported = {"target_std_ddof": 0, "target_scale_floor": 1e-8, "constant_target_scale": 1,
                 "unknown_categories": "impute_training_mode", "missing_indicators": "training_missing_only",
                 "all_constant_features": "one_zero_column"}
    if value["preprocessing"] != supported:
        raise ValueError("Preprocessing policy changed; implement and verify it before launching")
    if value["training"]["optimizer"] != "AdamW" or value["training"]["prior"] != "upstream_rand_dataset_filtered":
        raise ValueError("Unsupported optimizer/prior")
    if value["evaluation"]["device"] != "cpu" or value["evaluation"]["dtype"] != "float32":
        raise ValueError("Matched evaluation is fixed to CPU float32")
    return value


def identity():
    """Paths are relative so host and Docker obtain the same identity."""
    files = ("config.json", "source/model_config.json", "source/model.py", "source/interface.py",
             "pretraining/prior.py", "benchmarks/ctr23_task_ids.csv", "benchmarks/ctr23_published_baselines.csv",
             "common.py", "model_runtime.py", "train_baseline.py", "evaluate.py", "report.py", "prepare.py")
    hashes = {name: digest(ROOT / name) for name in files}
    return {"files": hashes, "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def parameter_shapes(model_config=None):
    """Shape algebra only; prepare/check verifies this against named_parameters()."""
    c = model_config or config()["model"]
    d, k = c["embed_dim"], c["embed_dim"] * c["n_cls_cols"]
    shapes = {}

    def linear(prefix, n_in, n_out):
        shapes[prefix + ".weight"] = [n_out, n_in]
        shapes[prefix + ".bias"] = [n_out]

    def norm(prefix, width):
        shapes[prefix + ".weight"] = [width]
        shapes[prefix + ".bias"] = [width]

    def mlp(prefix, n_in, n_hidden, n_out):
        linear(prefix + ".0", n_in, n_hidden)
        linear(prefix + ".2", n_hidden, n_out)

    def block(prefix, width, heads, ssmax):
        if width % heads or (width // heads) % 2:
            raise ValueError("Attention dimensions must be divisible with an even head width")
        shapes[prefix + ".in_proj_weight"] = [3 * width, width]
        shapes[prefix + ".in_proj_bias"] = [3 * width]
        linear(prefix + ".out_proj", width, width)
        mlp(prefix + ".mlp", width, 2 * width, width)
        norm(prefix + ".ln_attn", width)
        norm(prefix + ".ln_mlp", width)
        if ssmax:
            mlp(prefix + ".ssmax_layer.base_mlp", 1, 64, width)
            mlp(prefix + ".ssmax_layer.query_mlp", width // heads, 64, width // heads)

    linear("x_embed", c["feature_group_size"], d)
    linear("y_embed_in", 1, d)
    linear("y_embed_icl", 1, k)
    shapes["row_cls_tokens"] = [1, 1, c["n_cls_cols"], d]
    for i in range(c["col_num_blocks"]):
        prefix = f"col_blocks.{i}"
        shapes[prefix + ".inducing_vectors"] = [1, c["n_cls_rows"], d]
        block(prefix + ".tfm1", d, c["col_nhead"], True)
        block(prefix + ".tfm2", d, c["col_nhead"], False)
    for i in range(c["row_num_blocks"]):
        block(f"row_blocks.{i}", d, c["row_nhead"], False)
    for i in range(c["icl_num_blocks"]):
        block(f"icl_blocks.{i}", k, c["icl_nhead"], True)
    norm("row_ln", d)
    norm("out_ln", k)
    mlp("out_mlp", k, 2 * k, c["out_dim"])
    return dict(sorted(shapes.items()))


def parameter_count(shapes=None):
    return sum(math.prod(shape) for shape in (shapes or parameter_shapes()).values())


def require_prepared():
    path = FROZEN / "manifest.json"
    if not path.exists():
        raise RuntimeError("Prepare the experiment first: python3 manage.py prepare")
    value = read_json(path)
    if not value.get("complete") or value.get("identity") != identity():
        raise RuntimeError("Experiment config/sources changed; prepare again before launching")
    if digest(FROZEN / "parameters.json") != value["parameter_manifest_sha256"]:
        raise RuntimeError("Parameter manifest checksum changed")
    if read_json(FROZEN / "parameters.json")["shapes"] != parameter_shapes():
        raise RuntimeError("Parameter manifest differs from the frozen architecture")
    if not value["tasks"]:
        raise RuntimeError("No eligible CTR23 tasks")
    for task in value["tasks"]:
        for split in task["splits"]:
            if digest(FROZEN / split["file"]) != split["sha256"]:
                raise RuntimeError("Frozen split indices changed")
    return value


def retry_download(operation, label, attempts=5):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            status = getattr(error, "status_code", None) or getattr(error, "code", None)
            status = status or getattr(getattr(error, "response", None), "status_code", None)
            match = re.search(r"Status code:\s*(\d{3})", str(error))
            status = status or (int(match[1]) if match else None)
            temporary = status in {408, 429, 500, 502, 503, 504}
            if status is None:
                temporary = isinstance(error, (TimeoutError, ConnectionError)) or (
                    type(error).__module__.startswith(("requests.", "urllib3.")) and
                    type(error).__name__ in {"Timeout", "ReadTimeout", "ConnectTimeout", "ConnectionError"})
            if not temporary or attempt + 1 == attempts:
                raise
            delay = 5 * 2 ** attempt
            print(f"{label}: retry {attempt + 2}/{attempts} in {delay}s: {error}", flush=True)
            time.sleep(delay)


def baseline_table():
    with (ROOT / "benchmarks/ctr23_published_baselines.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    table = {}
    for row in rows:
        # The authors' misleading task_id column contains dataset names.
        key = re.sub(r"[^a-z0-9]", "", row["task_id"].lower())
        item = table.setdefault(key, {"dataset": row["task_id"], "scores": {}})
        if row["learner_id"] in item["scores"]:
            raise ValueError("Duplicate published baseline row")
        item["scores"][row["learner_id"]] = float(row["rmse"])
    return table


def validate_wsp_run(path):
    path = Path(path).resolve()
    if read_json(path / "state.json")["phase"] != "construction_finished":
        raise RuntimeError("WSP construction has not finished")
    metadata = read_json(path / "protocol.json")
    if metadata.get("experiment_identity") != identity() or metadata.get("model_config") != config()["model"]:
        raise RuntimeError("WSP run belongs to a different experiment")
    if read_json(path / "control/shapes.json") != parameter_shapes():
        raise RuntimeError("WSP parameter manifest changed")
    if set(metadata["source_sha256"]) != set(CONSTRUCTION_FILES):
        raise RuntimeError("Unexpected construction source inventory")
    for filename, sha in metadata["source_sha256"].items():
        if digest(path / "control/source" / filename) != sha or digest(ROOT / "source" / filename) != sha:
            raise RuntimeError("WSP source changed: " + filename)
    shapes = parameter_shapes()
    files = {}
    if (path / "work/weights").is_symlink():
        raise RuntimeError("WSP weights directory cannot be a symlink")
    for p in (path / "work/weights").glob("*.npy"):
        if p.is_symlink() or p.stem not in shapes:
            raise RuntimeError("Unexpected WSP tensor file")
        files[p.name] = digest(p)
    return {"path": str(path), "files": files, "audit_status": read_json(path / "state.json").get("audit_status", "not recorded")}


def validate_baseline_run(path):
    path = Path(path).resolve()
    state = read_json(path / "state.json")
    if state.get("phase") != "pretraining_finished" or state.get("kind") != "baseline":
        raise RuntimeError("The full baseline pretraining run has not finished")
    if state["experiment_identity"] != identity() or state["step"] != config()["training"]["steps"]:
        raise RuntimeError("Baseline configuration/budget differs from this experiment")
    if digest(path / "final.pt") != state["checkpoint_sha256"]:
        raise RuntimeError("Baseline checkpoint changed")
    return {"path": str(path), "sha256": state["checkpoint_sha256"], "step": state["step"]}
