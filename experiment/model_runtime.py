"""Scientific runtime; never exposed to the WSP sandbox."""
import importlib.metadata
from pathlib import Path
import numpy as np
import torch
import common
from source.model import NanoTabICLv2
from source.interface import Regressor


def packages():
    result = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "pandas", "openml"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def make_model(zero=False, model_config=None):
    cfg = model_config or common.config()["model"]
    model = NanoTabICLv2(**cfg).float()
    actual = {name: list(p.shape) for name, p in model.named_parameters()}
    if actual != common.parameter_shapes(cfg):
        raise RuntimeError("Shape algebra differs from the actual pinned upstream model")
    if zero:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        # Keep inv_freq and any other deterministic architecture buffers intact.
    return model


def load_wsp(directory, files):
    directory = Path(directory)
    if {p.name for p in directory.glob("*.npy")} != set(files):
        raise RuntimeError("WSP tensor filenames differ from the frozen snapshot")
    model = make_model(zero=True)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for filename, sha in files.items():
            path = directory / filename
            if common.digest(path) != sha:
                raise RuntimeError("WSP tensor digest changed")
            name = Path(filename).stem
            array = np.load(path, allow_pickle=False)
            if array.dtype != np.float32 or list(array.shape) != list(parameters[name].shape) or not np.isfinite(array).all():
                raise ValueError("Invalid WSP tensor: " + filename)
            parameters[name].copy_(torch.from_numpy(array.copy()))
    return model.cpu().eval()


def load_baseline(path, sha):
    if common.digest(path) != sha:
        raise RuntimeError("Pretrained checkpoint digest changed")
    # final.pt contains only tensors and simple metadata, not optimizer/RNG objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["kind"] != "baseline" or checkpoint["experiment_identity"] != common.identity() or
        checkpoint["model_config"] != common.config()["model"] or
        checkpoint["step"] != common.config()["training"]["steps"]):
        raise RuntimeError("Pretrained checkpoint does not match the frozen experiment")
    model = make_model()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise ValueError("Non-finite pretrained parameter")
    return model.cpu().float().eval()


def regression_scores(target, prediction, train_std):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.ndim != 1 or prediction.shape != target.shape or not len(target):
        raise ValueError("Prediction/target shapes differ")
    if not (np.isfinite(target).all() and np.isfinite(prediction).all()):
        raise ValueError("Non-finite target/prediction")
    mse = float(np.mean((prediction - target) ** 2))
    variance = float(np.mean((target - target.mean()) ** 2))
    rmse = float(np.sqrt(mse))
    if not np.isfinite(rmse):
        raise ValueError("Metric overflow")
    return {"rmse": rmse, "mae": float(np.mean(np.abs(target - prediction))),
            "r2": 1 - mse / variance if variance > 0 else None,
            "normalized_rmse": rmse / train_std if train_std > 1e-8 else None,
            "constant_training_target": train_std <= 1e-8}
