"""Leakage-safe model selection and evaluation helpers.

The public metrics in this project come from an outer evaluation loop.  Every
outer fold performs its own hyperparameter search using only the corresponding
outer-training rows.  Scaling and feature weighting are also refit within every
inner and outer training split.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


WeightFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class FittedSVR:
    model: SVR
    scaler_X: MinMaxScaler
    scaler_y: MinMaxScaler
    feature_weights: np.ndarray
    kernel_weights: np.ndarray


def parameter_combinations(grid: dict[str, Iterable]) -> list[dict[str, object]]:
    """Expand a small deterministic parameter grid."""
    keys = ("C", "gamma", "epsilon")
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def _splitter(n_samples: int, requested_splits: int, random_state: int) -> KFold:
    n_splits = min(requested_splits, n_samples)
    if n_splits < 2:
        raise ValueError("At least two samples are required for cross-validation.")
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _normalise_weights(weights: np.ndarray, n_features: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float).reshape(-1)
    if len(values) != n_features or not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError("Feature-weight function returned invalid weights.")
    values = np.clip(values, 1e-12, None)
    return values / values.sum()


def fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, object],
    weight_fn: WeightFunction,
    weighted: bool,
) -> FittedSVR:
    """Fit preprocessing, optional weights, and SVR on training rows only."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    n_features = X_train.shape[1]

    feature_weights = (
        _normalise_weights(weight_fn(X_train, y_train), n_features)
        if weighted
        else np.ones(n_features, dtype=float) / n_features
    )
    kernel_weights = np.sqrt(feature_weights) if weighted else np.ones(n_features)

    X_log = np.log(np.clip(X_train, 1e-3, None))
    y_log = np.log1p(y_train)
    scaler_X = MinMaxScaler().fit(X_log)
    scaler_y = MinMaxScaler().fit(y_log.reshape(-1, 1))
    X_weighted = scaler_X.transform(X_log) * kernel_weights
    y_scaled = scaler_y.transform(y_log.reshape(-1, 1)).ravel()

    model = SVR(kernel="rbf", **params)
    model.fit(X_weighted, y_scaled)
    return FittedSVR(model, scaler_X, scaler_y, feature_weights, kernel_weights)


def predict_model(bundle: FittedSVR, X_test: np.ndarray) -> np.ndarray:
    X_log = np.log(np.clip(np.asarray(X_test, dtype=float), 1e-3, None))
    X_weighted = bundle.scaler_X.transform(X_log) * bundle.kernel_weights
    prediction_scaled = bundle.model.predict(X_weighted)
    prediction_log = bundle.scaler_y.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).ravel()
    return np.maximum(np.expm1(prediction_log), 0.1)


def relative_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Signed relative error: (prediction - actual) / actual."""
    y_true = np.asarray(y_true, dtype=float)
    return (np.asarray(y_pred, dtype=float) - y_true) / y_true


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    signed = relative_errors(y_true, y_pred)
    absolute = np.abs(signed)
    return {
        "mmre": float(np.mean(absolute)),
        "pred25": float(np.mean(absolute <= 0.25)),
        "pred50": float(np.mean(absolute <= 0.50)),
        "pi_q10": float(np.percentile(signed, 10)),
        "pi_q90": float(np.percentile(signed, 90)),
    }


def prediction_interval(
    prediction: float, q10: float, q90: float, minimum: float = 0.1
) -> tuple[float, float]:
    """Convert signed relative-error quantiles into an actual-effort interval.

    With e = (prediction - actual) / actual, actual = prediction / (1 + e).
    Since that transform is decreasing, q90 defines the lower endpoint and q10
    defines the upper endpoint.
    """
    if q10 <= -1 or q90 <= -1 or q10 > q90:
        raise ValueError("Residual quantiles cannot produce a valid interval.")
    lower = max(minimum, float(prediction) / (1.0 + float(q90)))
    upper = max(lower, float(prediction) / (1.0 + float(q10)))
    return lower, upper


def _prepared_folds(
    X: np.ndarray,
    y: np.ndarray,
    weight_fn: WeightFunction,
    weighted: bool,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, np.ndarray]]:
    folds = []
    for train_idx, valid_idx in _splitter(len(y), n_splits, random_state).split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        n_features = X.shape[1]
        feature_weights = (
            _normalise_weights(weight_fn(X_train, y_train), n_features)
            if weighted
            else np.ones(n_features, dtype=float) / n_features
        )
        kernel_weights = np.sqrt(feature_weights) if weighted else np.ones(n_features)
        X_log = np.log(np.clip(X_train, 1e-3, None))
        y_log = np.log1p(y_train)
        scaler_X = MinMaxScaler().fit(X_log)
        scaler_y = MinMaxScaler().fit(y_log.reshape(-1, 1))
        folds.append(
            (
                scaler_X.transform(X_log) * kernel_weights,
                scaler_y.transform(y_log.reshape(-1, 1)).ravel(),
                scaler_X.transform(np.log(np.clip(X[valid_idx], 1e-3, None)))
                * kernel_weights,
                valid_idx,
                scaler_y,
                y[valid_idx],
            )
        )
    return folds


def select_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    weight_fn: WeightFunction,
    weighted: bool,
    param_grid: dict[str, Iterable],
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[dict[str, object], float]:
    """Select hyperparameters using CV confined to the supplied training rows."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    prepared = _prepared_folds(X, y, weight_fn, weighted, n_splits, random_state)
    best_params: dict[str, object] | None = None
    best_score = float("inf")

    for params in parameter_combinations(param_grid):
        predictions = np.empty(len(y), dtype=float)
        for X_train, y_train, X_valid, valid_idx, scaler_y, _ in prepared:
            model = SVR(kernel="rbf", **params)
            model.fit(X_train, y_train)
            pred_scaled = model.predict(X_valid)
            pred_log = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            predictions[valid_idx] = np.maximum(np.expm1(pred_log), 0.1)
        score = metrics_from_predictions(y, predictions)["mmre"]
        if score < best_score - 1e-12:
            best_score = score
            best_params = params

    if best_params is None:
        raise RuntimeError("Hyperparameter grid was empty.")
    return best_params, best_score


def nested_cv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    weight_fn: WeightFunction,
    weighted: bool,
    param_grid: dict[str, Iterable],
    outer_splits: int = 10,
    inner_splits: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    """Return pure outer-fold predictions from nested cross-validation."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    predictions = np.empty(len(y), dtype=float)
    fold_details = []

    splitter = _splitter(len(y), outer_splits, random_state)
    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        params, inner_mmre = select_hyperparameters(
            X[train_idx],
            y[train_idx],
            weight_fn,
            weighted,
            param_grid,
            n_splits=inner_splits,
            random_state=random_state + fold_number,
        )
        bundle = fit_model(X[train_idx], y[train_idx], params, weight_fn, weighted)
        predictions[test_idx] = predict_model(bundle, X[test_idx])
        fold_details.append(
            {
                "fold": fold_number,
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "inner_mmre": float(inner_mmre),
                "params": params,
            }
        )

    result = metrics_from_predictions(y, predictions)
    result.update(
        {
            "predictions": predictions,
            "signed_relative_errors": relative_errors(y, predictions),
            "fold_details": fold_details,
            "outer_splits": min(outer_splits, len(y)),
            "inner_splits": min(inner_splits, max(2, len(y) - len(y) // outer_splits)),
        }
    )
    return result


def fit_deployment_model(
    X: np.ndarray,
    y: np.ndarray,
    weight_fn: WeightFunction,
    weighted: bool,
    param_grid: dict[str, Iterable],
    inner_splits: int = 5,
    random_state: int = 142,
) -> tuple[FittedSVR, dict[str, object], float]:
    """Choose deployment parameters on all rows, then fit the deployed model."""
    params, selection_mmre = select_hyperparameters(
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float),
        weight_fn,
        weighted,
        param_grid,
        n_splits=inner_splits,
        random_state=random_state,
    )
    return fit_model(X, y, params, weight_fn, weighted), params, selection_mmre


def holdout_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    weight_fn: WeightFunction,
    weighted: bool,
    param_grid: dict[str, Iterable],
    inner_splits: int = 5,
    random_state: int = 242,
) -> dict[str, object]:
    """Evaluate one pure hold-out after tuning only on its training partition."""
    params, inner_mmre = select_hyperparameters(
        X_train,
        y_train,
        weight_fn,
        weighted,
        param_grid,
        n_splits=inner_splits,
        random_state=random_state,
    )
    bundle = fit_model(X_train, y_train, params, weight_fn, weighted)
    predictions = predict_model(bundle, X_test)
    result = metrics_from_predictions(y_test, predictions)
    result.update(
        {
            "predictions": predictions,
            "params": params,
            "inner_selection_mmre": inner_mmre,
        }
    )
    return result
