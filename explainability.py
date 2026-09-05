"""Leakage-safe SHAP research metrics for the weighted estimator.

The deployed app can explain its final full-data model, but those explanations
are not paper evidence.  This module instead fits models inside repeated nested
outer folds and explains only rows held out by each outer model.  Every SHAP
background is sampled only from that fold's training partition.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Callable, Iterable

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import KFold

from validation import (
    WeightFunction,
    fit_model,
    metrics_from_predictions,
    predict_model,
    select_hyperparameters,
)


def normalise_importance(values: np.ndarray) -> np.ndarray:
    """Return a finite non-negative importance distribution."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Importance values must be finite and non-empty.")
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0:
        raise ValueError("At least one importance value must be positive.")
    return array / total


def shap_reliance(shap_values: np.ndarray) -> np.ndarray:
    """Global model reliance: normalized mean absolute SHAP magnitude."""
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("SHAP values must have shape (projects, features).")
    return normalise_importance(np.mean(np.abs(values), axis=0))


def _safe_statistic(value) -> float:
    result = float(value)
    return result if np.isfinite(result) else 0.0


def agreement_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    top_k: int = 3,
) -> dict[str, float | int]:
    """Compare two feature-importance distributions.

    Spearman and Kendall measure rank agreement, Jensen-Shannon similarity
    measures magnitude agreement, and top-k scores expose agreement among the
    most prominent features.  The Kuncheva index corrects top-k overlap for
    agreement expected by chance.
    """
    p = normalise_importance(reference)
    q = normalise_importance(candidate)
    if len(p) != len(q):
        raise ValueError("Importance vectors must have the same length.")
    feature_count = len(p)
    k = min(max(int(top_k), 1), feature_count)
    p_top = set(np.argsort(p)[-k:])
    q_top = set(np.argsort(q)[-k:])
    intersection = len(p_top & q_top)
    union = len(p_top | q_top)
    if k == feature_count:
        kuncheva = 1.0
    else:
        kuncheva = (intersection * feature_count - k * k) / (k * (feature_count - k))
    js_distance = float(jensenshannon(p, q, base=2.0))
    return {
        "spearman_rho": _safe_statistic(spearmanr(p, q).statistic),
        "kendall_tau_b": _safe_statistic(kendalltau(p, q, variant="b").statistic),
        "jensen_shannon_similarity": float(1.0 - js_distance),
        "top_k": k,
        "top_k_intersection": intersection,
        "top_k_jaccard": float(intersection / union),
        "top_k_kuncheva": float(kuncheva),
    }


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"median": 0.0, "q25": 0.0, "q75": 0.0, "minimum": 0.0, "maximum": 0.0}
    return {
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def project_bootstrap_agreement(
    reference_runs: np.ndarray,
    shap_runs: np.ndarray,
    top_k: int = 3,
    n_resamples: int = 2000,
    random_state: int = 0,
) -> dict[str, object]:
    """Project-level percentile intervals for global agreement statistics.

    Explanations and fold-specific reference weights are first averaged across
    repeat models for each project. Projects, rather than individual SHAP
    values or overlapping folds, are then resampled. The resulting intervals
    are conditional on the fitted repeated-CV models and are therefore
    uncertainty summaries, not population-level causal confidence intervals.
    """
    references = np.asarray(reference_runs, dtype=float)
    explanations = np.asarray(shap_runs, dtype=float)
    if references.ndim != 3 or explanations.ndim != 3:
        raise ValueError("Expected arrays with shape (repeats, projects, features).")
    if references.shape != explanations.shape:
        raise ValueError("Reference and SHAP arrays must have identical shapes.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")

    reference_by_project = np.mean(references, axis=0)
    reliance_by_project = np.mean(np.abs(explanations), axis=0)
    project_count = reference_by_project.shape[0]
    rng = np.random.default_rng(random_state)
    tracked = {
        "spearman_rho": [],
        "kendall_tau_b": [],
        "jensen_shannon_similarity": [],
        "top_k_jaccard": [],
        "top_k_kuncheva": [],
    }
    for _ in range(n_resamples):
        sample = rng.integers(0, project_count, size=project_count)
        metrics = agreement_metrics(
            np.mean(reference_by_project[sample], axis=0),
            np.mean(reliance_by_project[sample], axis=0),
            top_k,
        )
        for name in tracked:
            tracked[name].append(float(metrics[name]))

    def interval(values: list[float]) -> dict[str, float]:
        return {
            "lower": float(np.percentile(values, 2.5)),
            "median": float(np.percentile(values, 50)),
            "upper": float(np.percentile(values, 97.5)),
        }

    return {
        "coverage": 0.95,
        "resamples": int(n_resamples),
        "resampling_unit": "COCOMO project after averaging its repeated out-of-fold explanations",
        "method": "project-level nonparametric percentile bootstrap, conditional on fitted repeated-CV models",
        "scope": "conditional project-resampling interval; model fitting, split selection, tuning, and explainer uncertainty are not re-run",
        "metrics": {name: interval(values) for name, values in tracked.items()},
    }


def permutation_null_agreement(
    reference: np.ndarray,
    candidate: np.ndarray,
    top_k: int = 3,
    n_permutations: int = 10_000,
    random_state: int = 0,
) -> dict[str, object]:
    """Compare observed agreement with random feature-label permutations.

    The null is deliberately narrow: the reference magnitudes are held fixed
    while their feature labels are permuted. It is a descriptive calibration
    benchmark, not a population-level hypothesis test.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    reference = normalise_importance(reference)
    candidate = normalise_importance(candidate)
    observed = agreement_metrics(reference, candidate, top_k)
    rng = np.random.default_rng(random_state)
    names = (
        "spearman_rho",
        "kendall_tau_b",
        "jensen_shannon_similarity",
        "top_k_jaccard",
        "top_k_kuncheva",
    )
    null = {name: [] for name in names}
    for _ in range(n_permutations):
        metrics = agreement_metrics(rng.permutation(reference), candidate, top_k)
        for name in names:
            null[name].append(float(metrics[name]))

    p_values = {}
    for name in names:
        observed_value = float(observed[name])
        values = np.asarray(null[name], dtype=float)
        if name in ("spearman_rho", "kendall_tau_b"):
            exceedances = np.count_nonzero(np.abs(values) >= abs(observed_value))
            tail = "two-sided absolute"
        else:
            exceedances = np.count_nonzero(values >= observed_value)
            tail = "upper"
        p_values[name] = {
            "empirical_p": float((exceedances + 1) / (n_permutations + 1)),
            "tail": tail,
        }
    return {
        "permutations": int(n_permutations),
        "random_state": int(random_state),
        "null": "fixed reference magnitudes with randomly permuted feature labels",
        "scope": "descriptive label-permutation benchmark; feature exchangeability is not assumed as a population model",
        "observed": observed,
        "null_summaries": {name: _summary(values) for name, values in null.items()},
        "empirical_p_values": p_values,
    }


def ranking_stability(vectors: np.ndarray, top_k: int = 3) -> dict[str, object]:
    """Summarize pairwise agreement between repeated global rankings."""
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("At least two repeated importance vectors are required.")
    comparisons = [agreement_metrics(values[a], values[b], top_k) for a, b in combinations(range(len(values)), 2)]
    return {
        "pair_count": len(comparisons),
        "spearman_rho": _summary(item["spearman_rho"] for item in comparisons),
        "kendall_tau_b": _summary(item["kendall_tau_b"] for item in comparisons),
        "top_k_jaccard": _summary(item["top_k_jaccard"] for item in comparisons),
        "top_k_kuncheva": _summary(item["top_k_kuncheva"] for item in comparisons),
    }


def local_explanation_stability(shap_runs: np.ndarray, top_k: int = 3) -> dict[str, object]:
    """Compare each project's signed local explanation across repeat models."""
    values = np.asarray(shap_runs, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("Expected SHAP values with shape (repeats, projects, features).")
    repeats, projects, features = values.shape
    k = min(max(int(top_k), 1), features)
    cosines: list[float] = []
    overlaps: list[float] = []
    sign_agreements: list[float] = []
    for project in range(projects):
        for first, second in combinations(range(repeats), 2):
            a = values[first, project]
            b = values[second, project]
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator > 0:
                cosines.append(float(np.dot(a, b) / denominator))
            a_top = set(np.argsort(np.abs(a))[-k:])
            b_top = set(np.argsort(np.abs(b))[-k:])
            overlaps.append(len(a_top & b_top) / len(a_top | b_top))
            active = (np.abs(a) > 1e-9) | (np.abs(b) > 1e-9)
            if np.any(active):
                sign_agreements.append(float(np.mean(np.sign(a[active]) == np.sign(b[active]))))
    return {
        "comparison_count": len(overlaps),
        "cosine_similarity": _summary(cosines),
        "top_k_jaccard": _summary(overlaps),
        "sign_agreement": _summary(sign_agreements),
    }


def kernel_shap_values(
    bundle,
    X_evaluation: np.ndarray,
    X_background: np.ndarray,
    nsamples: int,
    random_state: int,
    l1_reg: str | float | bool = 0,
) -> tuple[np.ndarray, float, float]:
    """Explain raw-input predictions in effort units with explicit regularization."""
    import shap

    evaluation = np.asarray(X_evaluation, dtype=float)
    background = np.asarray(X_background, dtype=float)
    state = np.random.get_state()
    try:
        np.random.seed(random_state)
        explainer = shap.KernelExplainer(lambda rows: predict_model(bundle, rows), background)
        values = np.asarray(
            explainer.shap_values(
                evaluation,
                nsamples=int(nsamples),
                l1_reg=l1_reg,
                silent=True,
            ),
            dtype=float,
        )
    finally:
        np.random.set_state(state)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    predictions = predict_model(bundle, evaluation)
    additivity_error = float(np.max(np.abs(predictions - (base_value + values.sum(axis=1)))))
    return values, base_value, additivity_error


def kernel_shap_numerical_sensitivity(
    bundle,
    X_evaluation: np.ndarray,
    X_background: np.ndarray,
    main_nsamples: int,
    top_k: int = 3,
    random_state: int = 0,
    sample_sizes: tuple[int, ...] = (128, 256, 512),
    reference_nsamples: int = 1024,
    seed_count: int = 3,
) -> dict[str, object]:
    """Audit KernelSHAP sampling and seed sensitivity on one fixed model/background.

    This is a numerical audit only. Evaluation rows may be in-sample because the
    objective is to isolate explainer variation while holding the fitted model,
    evaluation rows, and background fixed.
    """
    evaluation = np.asarray(X_evaluation, dtype=float)
    background = np.asarray(X_background, dtype=float)
    if len(evaluation) < 1 or len(background) < 1:
        raise ValueError("Sensitivity analysis needs evaluation and background rows.")

    reference_values, _, reference_error = kernel_shap_values(
        bundle,
        evaluation,
        background,
        nsamples=reference_nsamples,
        random_state=random_state,
        l1_reg=0,
    )
    reference_reliance = shap_reliance(reference_values)

    def local_cosine(candidate: np.ndarray) -> dict[str, float]:
        values = []
        for reference_row, candidate_row in zip(reference_values, candidate):
            denominator = float(
                np.linalg.norm(reference_row) * np.linalg.norm(candidate_row)
            )
            if denominator > 0:
                values.append(float(np.dot(reference_row, candidate_row) / denominator))
        return _summary(values)

    sample_results = []
    for nsamples in sorted(set(int(value) for value in sample_sizes)):
        values, _, error = kernel_shap_values(
            bundle,
            evaluation,
            background,
            nsamples=nsamples,
            random_state=random_state,
            l1_reg=0,
        )
        sample_results.append(
            {
                "nsamples": nsamples,
                "global_agreement_to_1024_reference": agreement_metrics(
                    reference_reliance, shap_reliance(values), top_k
                ),
                "local_cosine_to_1024_reference": local_cosine(values),
                "nonzero_features_per_explanation": _summary(
                    np.count_nonzero(np.abs(values) > 1e-10, axis=1)
                ),
                "max_additivity_error_person_months": float(error),
            }
        )

    seed_values = []
    seed_details = []
    for offset in range(seed_count):
        seed = random_state + offset
        values, _, error = kernel_shap_values(
            bundle,
            evaluation,
            background,
            nsamples=main_nsamples,
            random_state=seed,
            l1_reg=0,
        )
        seed_values.append(values)
        seed_details.append(
            {
                "seed": seed,
                "nonzero_features_per_explanation": _summary(
                    np.count_nonzero(np.abs(values) > 1e-10, axis=1)
                ),
                "max_additivity_error_person_months": float(error),
            }
        )
    seed_array = np.asarray(seed_values)
    seed_reliance = np.asarray([shap_reliance(values) for values in seed_array])
    return _round_mapping(
        {
            "purpose": "numerical explainer sensitivity with model, rows, and background held fixed",
            "l1_reg": 0,
            "feature_selection": "disabled; all features enter the weighted least-squares solution",
            "evaluation_rows": int(len(evaluation)),
            "background_rows": int(len(background)),
            "reference_nsamples": int(reference_nsamples),
            "reference_max_additivity_error_person_months": reference_error,
            "sample_size_results": sample_results,
            "seed_results": seed_details,
            "seed_global_ranking_stability": ranking_stability(seed_reliance, top_k),
            "seed_local_stability": local_explanation_stability(seed_array, top_k),
            "scope": "in-sample numerical audit only; it is not predictive-performance or feature-reliance evidence",
        }
    )


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _round_mapping(mapping: dict[str, object]) -> dict[str, object]:
    rounded: dict[str, object] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            rounded[key] = _round_mapping(value)
        elif isinstance(value, (float, np.floating)):
            rounded[key] = _rounded(value)
        elif isinstance(value, (int, np.integer)):
            rounded[key] = int(value)
        else:
            rounded[key] = value
    return rounded


def repeated_oof_shap_analysis(
    X: np.ndarray,
    y: np.ndarray,
    weight_fn: WeightFunction,
    param_grid: dict[str, Iterable],
    feature_names: list[str],
    repeats: int = 10,
    outer_splits: int = 5,
    inner_splits: int = 3,
    background_size: int | None = None,
    nsamples: int = 256,
    top_k: int = 3,
    random_state: int = 420,
    bootstrap_resamples: int = 2000,
    null_permutations: int = 10_000,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run repeated nested outer-fold SHAP analysis for weighted/unweighted SVR.

    The primary proxy-versus-reliance comparison uses SHAP from the unweighted
    model. Alignment with the weighted model is secondary because injecting the
    weights can mechanically increase agreement with those same weights.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X and y have incompatible shapes.")
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names must match the columns in X.")
    if repeats < 2:
        raise ValueError("At least two repeats are needed to estimate stability.")

    def index_manifest(indices: np.ndarray) -> dict[str, object]:
        values = np.asarray(indices, dtype=np.int64)
        return {
            "indices": [int(value) for value in values],
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }

    weighted_runs: list[np.ndarray] = []
    unweighted_runs: list[np.ndarray] = []
    project_prior_runs: list[np.ndarray] = []
    prior_runs: list[np.ndarray] = []
    weighted_reliance_runs: list[np.ndarray] = []
    unweighted_reliance_runs: list[np.ndarray] = []
    repeat_details: list[dict[str, object]] = []
    fold_reliance_weighted: list[np.ndarray] = []
    fold_reliance_unweighted: list[np.ndarray] = []
    max_additivity_error = 0.0
    additivity_errors: list[float] = []
    oof_records: list[dict[str, object]] = []

    for repeat in range(repeats):
        split_seed = random_state + repeat * 1000
        splitter = KFold(n_splits=min(outer_splits, len(y)), shuffle=True, random_state=split_seed)
        weighted_shap = np.zeros_like(X, dtype=float)
        unweighted_shap = np.zeros_like(X, dtype=float)
        weighted_predictions = np.zeros(len(y), dtype=float)
        unweighted_predictions = np.zeros(len(y), dtype=float)
        project_priors = np.zeros_like(X, dtype=float)
        fold_details: list[dict[str, object]] = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            X_train, y_train = X[train_idx], y[train_idx]
            prior = normalise_importance(weight_fn(X_train, y_train))
            project_priors[test_idx] = prior

            unweighted_params, _ = select_hyperparameters(
                X_train,
                y_train,
                weight_fn,
                False,
                param_grid,
                n_splits=inner_splits,
                random_state=split_seed + fold,
            )
            weighted_params, _ = select_hyperparameters(
                X_train,
                y_train,
                weight_fn,
                True,
                param_grid,
                n_splits=inner_splits,
                random_state=split_seed + fold,
            )
            unweighted_bundle = fit_model(X_train, y_train, unweighted_params, weight_fn, False)
            weighted_bundle = fit_model(X_train, y_train, weighted_params, weight_fn, True)
            unweighted_predictions[test_idx] = predict_model(unweighted_bundle, X[test_idx])
            weighted_predictions[test_idx] = predict_model(weighted_bundle, X[test_idx])

            if background_size is None or background_size >= len(train_idx):
                background = X_train
                background_idx = train_idx
            else:
                rng = np.random.default_rng(split_seed + fold * 17)
                chosen = rng.choice(len(train_idx), size=background_size, replace=False)
                background = X_train[chosen]
                background_idx = train_idx[chosen]
            shap_seed = split_seed + fold * 31
            fold_unweighted, base_unweighted, error_unweighted = kernel_shap_values(
                unweighted_bundle, X[test_idx], background, nsamples, shap_seed, l1_reg=0
            )
            fold_weighted, base_weighted, error_weighted = kernel_shap_values(
                weighted_bundle, X[test_idx], background, nsamples, shap_seed, l1_reg=0
            )
            unweighted_shap[test_idx] = fold_unweighted
            weighted_shap[test_idx] = fold_weighted
            max_additivity_error = max(max_additivity_error, error_unweighted, error_weighted)
            additivity_errors.extend([error_unweighted, error_weighted])

            for local_position, row_index in enumerate(test_idx):
                for feature_index, feature_name in enumerate(feature_names):
                    oof_records.append(
                        {
                            "repeat": repeat + 1,
                            "fold": fold,
                            "row_index": int(row_index),
                            "feature": feature_name,
                            "weighting_proxy": float(prior[feature_index]),
                            "unweighted_shap_pm": float(fold_unweighted[local_position, feature_index]),
                            "weighted_shap_pm": float(fold_weighted[local_position, feature_index]),
                            "unweighted_base_pm": base_unweighted,
                            "weighted_base_pm": base_weighted,
                            "unweighted_prediction_pm": float(unweighted_predictions[row_index]),
                            "weighted_prediction_pm": float(weighted_predictions[row_index]),
                            "actual_effort_pm": float(y[row_index]),
                            "shap_nsamples": int(nsamples),
                            "shap_l1_reg": 0,
                        }
                    )

            unweighted_reliance = shap_reliance(fold_unweighted)
            weighted_reliance = shap_reliance(fold_weighted)
            fold_reliance_unweighted.append(unweighted_reliance)
            fold_reliance_weighted.append(weighted_reliance)
            fold_details.append(
                {
                    "fold": fold,
                    "train_n": int(len(train_idx)),
                    "test_n": int(len(test_idx)),
                    "train_manifest": index_manifest(train_idx),
                    "test_manifest": index_manifest(test_idx),
                    "background_manifest": index_manifest(background_idx),
                    "unweighted_params": unweighted_params,
                    "weighted_params": weighted_params,
                    "primary_unweighted_alignment": agreement_metrics(prior, unweighted_reliance, top_k),
                    "weighted_alignment_descriptive": agreement_metrics(prior, weighted_reliance, top_k),
                }
            )

        repeat_prior = normalise_importance(np.mean(project_priors, axis=0))
        repeat_unweighted_reliance = shap_reliance(unweighted_shap)
        repeat_weighted_reliance = shap_reliance(weighted_shap)
        prior_runs.append(repeat_prior)
        project_prior_runs.append(project_priors)
        unweighted_runs.append(unweighted_shap)
        weighted_runs.append(weighted_shap)
        unweighted_reliance_runs.append(repeat_unweighted_reliance)
        weighted_reliance_runs.append(repeat_weighted_reliance)
        repeat_details.append(
            {
                "repeat": repeat + 1,
                "weighted_accuracy": metrics_from_predictions(y, weighted_predictions),
                "unweighted_accuracy": metrics_from_predictions(y, unweighted_predictions),
                "primary_unweighted_alignment": agreement_metrics(
                    repeat_prior, repeat_unweighted_reliance, top_k
                ),
                "weighted_alignment_descriptive": agreement_metrics(
                    repeat_prior, repeat_weighted_reliance, top_k
                ),
                "folds": fold_details,
            }
        )

    prior_array = np.asarray(prior_runs)
    project_prior_array = np.asarray(project_prior_runs)
    unweighted_array = np.asarray(unweighted_runs)
    weighted_array = np.asarray(weighted_runs)
    unweighted_reliance_array = np.asarray(unweighted_reliance_runs)
    weighted_reliance_array = np.asarray(weighted_reliance_runs)
    global_prior = normalise_importance(np.mean(prior_array, axis=0))
    global_unweighted = normalise_importance(np.mean(unweighted_reliance_array, axis=0))
    global_weighted = normalise_importance(np.mean(weighted_reliance_array, axis=0))

    def top_frequency(vectors: np.ndarray) -> np.ndarray:
        counts = np.zeros(X.shape[1], dtype=float)
        for vector in vectors:
            counts[np.argsort(vector)[-min(top_k, X.shape[1]):]] += 1
        return counts / len(vectors)

    unweighted_frequency = top_frequency(unweighted_reliance_array)
    weighted_frequency = top_frequency(weighted_reliance_array)
    prior_rank = np.argsort(np.argsort(-global_prior)) + 1
    unweighted_rank = np.argsort(np.argsort(-global_unweighted)) + 1
    weighted_rank = np.argsort(np.argsort(-global_weighted)) + 1
    prior_ranks_by_repeat = np.asarray([np.argsort(np.argsort(-row)) + 1 for row in prior_array])
    unweighted_ranks_by_repeat = np.asarray(
        [np.argsort(np.argsort(-row)) + 1 for row in unweighted_reliance_array]
    )
    weighted_ranks_by_repeat = np.asarray(
        [np.argsort(np.argsort(-row)) + 1 for row in weighted_reliance_array]
    )
    feature_table = [
        {
            "feature": feature_names[index],
            "mean_weighting_proxy": _rounded(global_prior[index]),
            "proxy_rank": int(prior_rank[index]),
            "proxy_rank_median": _rounded(np.median(prior_ranks_by_repeat[:, index])),
            "proxy_rank_iqr": [
                _rounded(np.percentile(prior_ranks_by_repeat[:, index], 25)),
                _rounded(np.percentile(prior_ranks_by_repeat[:, index], 75)),
            ],
            # Compatibility aliases retained for the existing app/document schema.
            "mean_fahp_prior": _rounded(global_prior[index]),
            "unweighted_oof_shap_reliance": _rounded(global_unweighted[index]),
            "weighted_oof_shap_reliance": _rounded(global_weighted[index]),
            "fahp_rank": int(prior_rank[index]),
            "unweighted_shap_rank": int(unweighted_rank[index]),
            "weighted_shap_rank": int(weighted_rank[index]),
            "fahp_rank_median": _rounded(np.median(prior_ranks_by_repeat[:, index])),
            "fahp_rank_iqr": [
                _rounded(np.percentile(prior_ranks_by_repeat[:, index], 25)),
                _rounded(np.percentile(prior_ranks_by_repeat[:, index], 75)),
            ],
            "unweighted_shap_rank_median": _rounded(
                np.median(unweighted_ranks_by_repeat[:, index])
            ),
            "unweighted_shap_rank_iqr": [
                _rounded(np.percentile(unweighted_ranks_by_repeat[:, index], 25)),
                _rounded(np.percentile(unweighted_ranks_by_repeat[:, index], 75)),
            ],
            "weighted_shap_rank_median": _rounded(
                np.median(weighted_ranks_by_repeat[:, index])
            ),
            "weighted_shap_rank_iqr": [
                _rounded(np.percentile(weighted_ranks_by_repeat[:, index], 25)),
                _rounded(np.percentile(weighted_ranks_by_repeat[:, index], 75)),
            ],
            "unweighted_top_k_frequency": _rounded(unweighted_frequency[index]),
            "weighted_top_k_frequency": _rounded(weighted_frequency[index]),
        }
        for index in np.argsort(global_unweighted)[::-1]
    ]

    result = {
        "status": "pilot research analysis",
        "protocol": {
            "repeats": repeats,
            "outer_splits": min(outer_splits, len(y)),
            "inner_splits": min(inner_splits, len(y) - 1),
            "explainer": "model-agnostic KernelSHAP over the complete raw-input prediction pipeline",
            "background_size": "all outer-training projects" if background_size is None else background_size,
            "background_source": "outer-training partition only; outcomes are not used to select background rows",
            "nsamples": nsamples,
            "explained_rows": "outer-test projects only; each project is explained once per repeat",
            "top_k": min(top_k, X.shape[1]),
            "random_state": random_state,
            "bootstrap_resamples": int(bootstrap_resamples),
            "null_permutations": int(null_permutations),
            "l1_reg": 0,
            "feature_selection": f"disabled; all {X.shape[1]} features enter the weighted least-squares solution",
            "fold_audit": "zero-based train/test/background row manifests and selected parameters are stored per fold",
        },
        "weighting_proxy": {
            "method": "Chang extent analysis on training-project effort/LOC, followed by absolute Spearman feature association",
            "target_informed": True,
            "expert_derived": False,
            "interpretation": "supervised FAHP-inspired proxy, not an independent expert prior",
            "legacy_feature_table_aliases": {
                "mean_fahp_prior": "mean_weighting_proxy",
                "fahp_rank": "proxy_rank",
            },
        },
        "primary_comparison": "training-derived weighting proxy versus unweighted-model out-of-fold SHAP reliance",
        "primary_alignment": agreement_metrics(global_prior, global_unweighted, top_k),
        "primary_alignment_project_bootstrap": project_bootstrap_agreement(
            project_prior_array,
            unweighted_array,
            top_k=top_k,
            n_resamples=bootstrap_resamples,
            random_state=random_state + 900_000,
        ),
        "weighted_alignment_descriptive": agreement_metrics(global_prior, global_weighted, top_k),
        "feature_label_permutation_null": permutation_null_agreement(
            global_prior,
            global_unweighted,
            top_k=top_k,
            n_permutations=null_permutations,
            random_state=random_state + 800_000,
        ),
        "primary_alignment_across_repeats": {
            metric: _summary(
                detail["primary_unweighted_alignment"][metric] for detail in repeat_details
            )
            for metric in (
                "spearman_rho", "kendall_tau_b", "jensen_shannon_similarity",
                "top_k_jaccard", "top_k_kuncheva",
            )
        },
        "weighted_alignment_across_repeats_descriptive": {
            metric: _summary(
                detail["weighted_alignment_descriptive"][metric] for detail in repeat_details
            )
            for metric in (
                "spearman_rho", "kendall_tau_b", "jensen_shannon_similarity",
                "top_k_jaccard", "top_k_kuncheva",
            )
        },
        "unweighted_global_ranking_stability": ranking_stability(unweighted_reliance_array, top_k),
        "weighted_global_ranking_stability": ranking_stability(weighted_reliance_array, top_k),
        "unweighted_local_stability": local_explanation_stability(unweighted_array, top_k),
        "weighted_local_stability": local_explanation_stability(weighted_array, top_k),
        "joint_stability_scope": "split, fitted model, outer-training background, and KernelSHAP seed vary together; this is joint resampling stability, not isolated model or explainer stability",
        "nonzero_features_per_explanation": {
            "unweighted": _summary(
                np.count_nonzero(np.abs(unweighted_array) > 1e-10, axis=2).ravel()
            ),
            "weighted": _summary(
                np.count_nonzero(np.abs(weighted_array) > 1e-10, axis=2).ravel()
            ),
            "threshold": "1e-10 absolute person-months",
        },
        "repeated_accuracy_descriptive": {
            "weighted_mmre": _summary(
                detail["weighted_accuracy"]["mmre"] for detail in repeat_details
            ),
            "unweighted_mmre": _summary(
                detail["unweighted_accuracy"]["mmre"] for detail in repeat_details
            ),
            "paired_mmre_difference_unweighted_minus_weighted": _summary(
                detail["unweighted_accuracy"]["mmre"]
                - detail["weighted_accuracy"]["mmre"]
                for detail in repeat_details
            ),
            "weighted_mdmre": _summary(
                detail["weighted_accuracy"]["mdmre"] for detail in repeat_details
            ),
            "unweighted_mdmre": _summary(
                detail["unweighted_accuracy"]["mdmre"] for detail in repeat_details
            ),
            "paired_mdmre_difference_unweighted_minus_weighted": _summary(
                detail["unweighted_accuracy"]["mdmre"]
                - detail["weighted_accuracy"]["mdmre"]
                for detail in repeat_details
            ),
            "weighted_mae_person_months": _summary(
                detail["weighted_accuracy"]["mae"] for detail in repeat_details
            ),
            "unweighted_mae_person_months": _summary(
                detail["unweighted_accuracy"]["mae"] for detail in repeat_details
            ),
            "paired_mae_difference_unweighted_minus_weighted_person_months": _summary(
                detail["unweighted_accuracy"]["mae"]
                - detail["weighted_accuracy"]["mae"]
                for detail in repeat_details
            ),
            "weighted_better_repeat_count": sum(
                detail["weighted_accuracy"]["mmre"]
                < detail["unweighted_accuracy"]["mmre"]
                for detail in repeat_details
            ),
            "repeat_count": repeats,
            "caveat": "Repeated partitions overlap and are not independent inferential units; no uncorrected significance test is reported.",
        },
        "fold_level_stability": {
            "unweighted": ranking_stability(np.asarray(fold_reliance_unweighted), top_k),
            "weighted": ranking_stability(np.asarray(fold_reliance_weighted), top_k),
            "caveat": "Outer folds contain few projects, fold test compositions differ, and the 1,225 fold pairs are dependent; repeat-level pooled rankings are the primary joint-stability result.",
        },
        "max_additivity_error_person_months": _rounded(max_additivity_error),
        "additivity_error_person_months": _summary(additivity_errors),
        "feature_table": feature_table,
        "repeat_details": repeat_details,
        "interpretation_limits": [
            "SHAP decomposes model predictions relative to a chosen training background; it is not causal evidence.",
            "Alignment with weighted-model SHAP is descriptive because the same weights are injected into that model.",
            "Current feature weights are data-derived FAHP-inspired proxies, not an independent panel of expert judgments.",
            "Correlated COCOMO drivers can redistribute attribution; dependence-aware sensitivity analysis remains future work.",
            "The conditional project-resampling interval fixes the fitted repeated-CV models and does not include every source of uncertainty.",
            "Joint repeat stability combines changes in split, fitted model, background, and KernelSHAP seed; numerical explainer sensitivity is reported separately.",
        ],
    }
    if max_additivity_error > 1e-5:
        raise RuntimeError(
            f"KernelSHAP additivity error {max_additivity_error:.6g} PM exceeds 1e-5 PM."
        )
    return _round_mapping(result), oof_records
