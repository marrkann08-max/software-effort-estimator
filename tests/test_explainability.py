import numpy as np
import pytest

import explainability
from explainability import (
    agreement_metrics,
    local_explanation_stability,
    normalise_importance,
    project_bootstrap_agreement,
    repeated_oof_shap_analysis,
    ranking_stability,
    shap_reliance,
)


def test_normalise_importance_rejects_invalid_vectors():
    assert normalise_importance(np.array([1.0, 2.0, 1.0])).tolist() == pytest.approx(
        [0.25, 0.5, 0.25]
    )
    with pytest.raises(ValueError):
        normalise_importance(np.zeros(3))
    with pytest.raises(ValueError):
        normalise_importance(np.array([1.0, np.nan]))


def test_shap_reliance_uses_mean_absolute_attribution():
    values = np.array([[2.0, -1.0, 0.0], [-2.0, 3.0, 0.0]])
    assert shap_reliance(values).tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_identical_rankings_have_perfect_agreement():
    values = np.array([0.6, 0.25, 0.1, 0.05])
    result = agreement_metrics(values, values, top_k=2)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["kendall_tau_b"] == pytest.approx(1.0)
    assert result["jensen_shannon_similarity"] == pytest.approx(1.0)
    assert result["top_k_jaccard"] == pytest.approx(1.0)
    assert result["top_k_kuncheva"] == pytest.approx(1.0)


def test_reversed_rankings_are_detected():
    first = np.array([0.4, 0.3, 0.2, 0.1])
    second = first[::-1]
    result = agreement_metrics(first, second, top_k=2)
    assert result["spearman_rho"] == pytest.approx(-1.0)
    assert result["top_k_intersection"] == 0
    assert result["top_k_jaccard"] == pytest.approx(0.0)
    assert result["top_k_kuncheva"] == pytest.approx(-1.0)


def test_global_and_local_stability_are_one_for_identical_repeats():
    rankings = np.array([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
    global_result = ranking_stability(rankings, top_k=2)
    assert global_result["spearman_rho"]["median"] == pytest.approx(1.0)
    assert global_result["top_k_jaccard"]["median"] == pytest.approx(1.0)

    one_run = np.array([[2.0, -1.0, 0.2], [-3.0, 0.5, 0.1]])
    local_result = local_explanation_stability(
        np.stack([one_run, one_run, one_run]), top_k=2
    )
    assert local_result["cosine_similarity"]["median"] == pytest.approx(1.0)
    assert local_result["top_k_jaccard"]["median"] == pytest.approx(1.0)
    assert local_result["sign_agreement"]["median"] == pytest.approx(1.0)


def test_project_bootstrap_resamples_projects_and_is_reproducible():
    references = np.array(
        [
            [[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]],
            [[0.75, 0.25], [0.65, 0.35], [0.85, 0.15]],
        ]
    )
    explanations = references.copy()
    first = project_bootstrap_agreement(
        references, explanations, top_k=1, n_resamples=50, random_state=9
    )
    second = project_bootstrap_agreement(
        references, explanations, top_k=1, n_resamples=50, random_state=9
    )

    assert first == second
    assert first["resamples"] == 50
    assert first["metrics"]["spearman_rho"]["lower"] == pytest.approx(1.0)


def test_repeated_oof_flow_keeps_test_rows_out_of_fit_and_background(monkeypatch):
    X = np.column_stack(
        [np.arange(8, dtype=float), np.arange(8, dtype=float) + 10, np.arange(8) + 20]
    )
    y = np.arange(8, dtype=float) + 2
    checked_calls = []

    class FakeBundle:
        def __init__(self, rows, weighted):
            self.training_ids = set(np.asarray(rows)[:, 0].astype(int))
            self.weighted = weighted

    def fake_select(X_train, _y_train, _weight_fn, _weighted, _grid, **_kwargs):
        assert 0 < len(X_train) < len(X)
        return {"C": 1, "gamma": 0.1, "epsilon": 0.1}, 0.1

    def fake_fit(X_train, _y_train, _params, _weight_fn, weighted):
        assert 0 < len(X_train) < len(X)
        return FakeBundle(X_train, weighted)

    def fake_predict(bundle, rows):
        row_ids = np.asarray(rows)[:, 0]
        return row_ids + (2.0 if bundle.weighted else 1.0)

    def fake_kernel(bundle, evaluation, background, nsamples, random_state, l1_reg):
        evaluation_ids = set(np.asarray(evaluation)[:, 0].astype(int))
        background_ids = set(np.asarray(background)[:, 0].astype(int))
        assert evaluation_ids.isdisjoint(bundle.training_ids)
        assert evaluation_ids.isdisjoint(background_ids)
        assert background_ids == bundle.training_ids
        assert nsamples == 32
        assert l1_reg == 0
        checked_calls.append((evaluation_ids, background_ids, random_state))
        values = np.tile(np.array([0.6, -0.3, 0.1]), (len(evaluation), 1))
        return values, 1.0, 0.0

    monkeypatch.setattr(explainability, "select_hyperparameters", fake_select)
    monkeypatch.setattr(explainability, "fit_model", fake_fit)
    monkeypatch.setattr(explainability, "predict_model", fake_predict)
    monkeypatch.setattr(explainability, "kernel_shap_values", fake_kernel)

    evidence, ledger = repeated_oof_shap_analysis(
        X,
        y,
        lambda rows, _targets: np.array([0.5, 0.3, 0.2]),
        {"C": [1], "gamma": [0.1], "epsilon": [0.1]},
        ["a", "b", "c"],
        repeats=2,
        outer_splits=2,
        inner_splits=2,
        nsamples=32,
        top_k=1,
        random_state=5,
        bootstrap_resamples=20,
        null_permutations=20,
    )

    assert len(checked_calls) == 2 * 2 * 2  # repeats × folds × model variants
    assert len(ledger) == 2 * len(X) * X.shape[1]
    for detail in evidence["repeat_details"]:
        for fold in detail["folds"]:
            train = set(fold["train_manifest"]["indices"])
            test = set(fold["test_manifest"]["indices"])
            background = set(fold["background_manifest"]["indices"])
            assert train.isdisjoint(test)
            assert background == train
