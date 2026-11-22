"""Fixed preprocessing shared by WSP evaluation and synthetic pretraining.

This file contains no data loading, prior generation, training or checkpoint IO.
X standardization is done by the unchanged upstream model on context rows.
"""
import numpy as np
import pandas as pd
import torch


class FeaturePreprocessor:
    def fit(self, X):
        X = pd.DataFrame(X)
        self.n_input = X.shape[1]
        self.columns = []
        for j in range(self.n_input):
            column = X.iloc[:, j]
            numeric = pd.to_numeric(column, errors="coerce")
            is_numeric = not isinstance(column.dtype, pd.CategoricalDtype) and numeric.notna().sum() == column.notna().sum()
            if is_numeric:
                values = numeric.to_numpy(dtype=float, na_value=np.nan, copy=True)
                values[~np.isfinite(values)] = np.nan
                finite = values[np.isfinite(values)]
                if len(np.unique(finite)) <= 1:
                    continue
                spec = {"position": j, "numeric": True, "fill": float(finite.mean()), "indicator": bool(np.isnan(values).any())}
            else:
                values = [None if pd.isna(v) else str(v) for v in column]
                categories = sorted(set(v for v in values if v is not None))
                if len(categories) <= 1:
                    continue
                mapping = {v: i for i, v in enumerate(categories)}
                codes = np.asarray([mapping.get(v, np.nan) for v in values], dtype=float)
                fill = int(np.bincount(codes[np.isfinite(codes)].astype(int)).argmax())
                spec = {"position": j, "numeric": False, "mapping": mapping, "fill": fill, "indicator": bool(np.isnan(codes).any())}
            self.columns.append(spec)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        if X.shape[1] != self.n_input:
            raise ValueError("Input column count differs from context")
        output = []
        for spec in self.columns:
            column = X.iloc[:, spec["position"]]
            if spec["numeric"]:
                values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)
            else:
                values = np.asarray([spec["mapping"].get(None if pd.isna(v) else str(v), np.nan) for v in column], dtype=float)
            missing = ~np.isfinite(values)
            output.append(np.where(missing, spec["fill"], values))
            if spec["indicator"]:
                output.append(missing.astype(float))
        if not output:
            return np.zeros((len(X), 1), dtype=np.float32)
        result = np.column_stack(output).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Feature preprocessing overflowed float32")
        return result


def target_stats(y):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Targets must be a nonempty finite vector")
    mean, std = float(y.mean()), float(y.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Context target statistics overflowed")
    scale = std if std > 1e-8 else 1.0
    return mean, scale, std


def context_tensors(X_context, y_context, X_query, device="cpu"):
    preprocessor = FeaturePreprocessor().fit(X_context)
    x_context = preprocessor.transform(X_context)
    x_query = preprocessor.transform(X_query)
    mean, scale, std = target_stats(y_context)
    x = torch.as_tensor(np.concatenate((x_context, x_query)), device=device).unsqueeze(0)
    y = torch.as_tensor((np.asarray(y_context) - mean) / scale, dtype=torch.float32, device=device).unsqueeze(0)
    return x, y, (mean, scale, std)


class Regressor:
    def __init__(self, model, device="cpu", test_batch_size=256):
        if test_batch_size < 1:
            raise ValueError("test_batch_size must be positive")
        self.model = model.to(device).float().eval()
        self.device = device
        self.test_batch_size = test_batch_size

    def fit(self, X, y):
        self.preprocessor = FeaturePreprocessor().fit(X)
        self.X = self.preprocessor.transform(X)
        self.mean, self.scale, self.std = target_stats(y)
        self.y = ((np.asarray(y) - self.mean) / self.scale).astype(np.float32)
        return self

    def predict(self, X):
        query = self.preprocessor.transform(X)
        result = []
        # Queries never attend to other queries in upstream nanoTabICL. Column
        # inducing vectors and all ICL keys/values depend on context rows only;
        # row RoPE acts across columns. Batching queries preserves the model's
        # computation while retaining every context row in every batch.
        with torch.no_grad():
            y = torch.as_tensor(self.y, device=self.device).unsqueeze(0)
            for start in range(0, len(query), self.test_batch_size):
                x = np.concatenate((self.X, query[start:start + self.test_batch_size]))
                x = torch.as_tensor(x, device=self.device).unsqueeze(0)
                pred = self.model(x, y).reshape(-1).detach().cpu().numpy().astype(float)
                result.append(pred * self.scale + self.mean)
        predictions = np.concatenate(result) if result else np.empty(0)
        if not np.isfinite(predictions).all():
            raise ValueError("Model produced non-finite predictions")
        return predictions
