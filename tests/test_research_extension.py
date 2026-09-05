import json
from pathlib import Path

import numpy as np
import pandas as pd

from explainability import agreement_metrics


ROOT = Path(__file__).resolve().parents[1]


def _explainability_metrics():
    metrics = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    return metrics["models"]["cocomo81"]["explainability"]


def test_explainability_protocol_and_feature_distributions_are_complete():
    evidence = _explainability_metrics()
    protocol = evidence["protocol"]
    assert protocol["repeats"] == 10
    assert protocol["outer_splits"] == 5
    assert protocol["inner_splits"] == 3
    assert protocol["nsamples"] == 256
    assert protocol["l1_reg"] == 0
    assert protocol["feature_selection"].startswith("disabled")
    assert protocol["background_size"] == "all outer-training projects"
    assert protocol["explained_rows"] == (
        "outer-test projects only; each project is explained once per repeat"
    )
    assert "unweighted-model" in evidence["primary_comparison"]

    rows = evidence["feature_table"]
    assert len(rows) == 17
    assert len({row["feature"] for row in rows}) == 17
    np.testing.assert_allclose(
        sum(row["mean_weighting_proxy"] for row in rows), 1.0, atol=1e-5
    )
    np.testing.assert_allclose(
        sum(row["unweighted_oof_shap_reliance"] for row in rows), 1.0, atol=1e-5
    )


def test_explainability_intervals_and_additivity_are_valid():
    evidence = _explainability_metrics()
    bootstrap = evidence["primary_alignment_project_bootstrap"]
    assert bootstrap["resamples"] == 2000
    assert bootstrap["coverage"] == 0.95
    assert "conditional" in bootstrap["scope"]
    assert bootstrap["resampling_unit"].startswith("COCOMO project")
    for interval in bootstrap["metrics"].values():
        assert interval["lower"] <= interval["median"] <= interval["upper"]
        assert np.isfinite(list(interval.values())).all()
    assert evidence["max_additivity_error_person_months"] <= 1e-5
    assert evidence["feature_label_permutation_null"]["permutations"] == 10_000
    assert evidence["nonzero_features_per_explanation"]["unweighted"]["median"] > 10

    numerical = evidence["numerical_sensitivity"]
    assert numerical["l1_reg"] == 0
    assert numerical["reference_nsamples"] == 1024
    assert {row["nsamples"] for row in numerical["sample_size_results"]} == {
        128,
        256,
        512,
    }


def test_oof_explanation_ledger_has_exact_coverage_and_additivity():
    ledger = pd.read_csv(ROOT / "explainability_oof.csv")
    required = {
        "repeat",
        "fold",
        "row_index",
        "feature",
        "weighting_proxy",
        "unweighted_shap_pm",
        "weighted_shap_pm",
        "unweighted_base_pm",
        "weighted_base_pm",
        "unweighted_prediction_pm",
        "weighted_prediction_pm",
        "actual_effort_pm",
    }
    assert required.issubset(ledger.columns)
    assert len(ledger) == 10 * 63 * 17
    assert ledger["repeat"].nunique() == 10
    assert ledger.groupby(["repeat", "row_index", "feature"]).size().eq(1).all()
    assert ledger.groupby(["repeat", "row_index"])["feature"].nunique().eq(17).all()

    grouped = ledger.groupby(["repeat", "row_index"], sort=False)
    unweighted_reconstructed = (
        grouped["unweighted_base_pm"].first()
        + grouped["unweighted_shap_pm"].sum()
    )
    weighted_reconstructed = (
        grouped["weighted_base_pm"].first()
        + grouped["weighted_shap_pm"].sum()
    )
    np.testing.assert_allclose(
        unweighted_reconstructed,
        grouped["unweighted_prediction_pm"].first(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        weighted_reconstructed,
        grouped["weighted_prediction_pm"].first(),
        atol=1e-5,
    )

    nonzero_counts = grouped["unweighted_shap_pm"].apply(
        lambda values: np.count_nonzero(np.abs(values.to_numpy()) > 1e-10)
    )
    assert (nonzero_counts > 10).mean() >= 0.95


def test_primary_agreement_recomputes_independently_from_oof_ledger():
    evidence = _explainability_metrics()
    ledger = pd.read_csv(ROOT / "explainability_oof.csv")
    proxy_by_repeat = (
        ledger.groupby(["repeat", "feature"])["weighting_proxy"]
        .mean()
        .unstack("feature")
    )
    shap_by_repeat = (
        ledger.assign(abs_shap=ledger["unweighted_shap_pm"].abs())
        .groupby(["repeat", "feature"])["abs_shap"]
        .mean()
        .unstack("feature")
    )
    proxy_by_repeat = proxy_by_repeat.div(proxy_by_repeat.sum(axis=1), axis=0)
    shap_by_repeat = shap_by_repeat.div(shap_by_repeat.sum(axis=1), axis=0)
    recomputed = agreement_metrics(
        proxy_by_repeat.mean(axis=0).to_numpy(),
        shap_by_repeat.mean(axis=0).to_numpy(),
        top_k=3,
    )
    for name in (
        "spearman_rho",
        "kendall_tau_b",
        "jensen_shannon_similarity",
        "top_k_jaccard",
        "top_k_kuncheva",
    ):
        np.testing.assert_allclose(
            recomputed[name], evidence["primary_alignment"][name], atol=1e-6
        )


def test_fold_manifests_prove_training_background_excludes_outer_test_rows():
    evidence = _explainability_metrics()
    for repeat in evidence["repeat_details"]:
        covered_test_rows = []
        for fold in repeat["folds"]:
            train = set(fold["train_manifest"]["indices"])
            test = set(fold["test_manifest"]["indices"])
            background = set(fold["background_manifest"]["indices"])
            assert train.isdisjoint(test)
            assert background.issubset(train)
            assert background.isdisjoint(test)
            assert fold["unweighted_params"]
            assert fold["weighted_params"]
            covered_test_rows.extend(test)
        assert sorted(covered_test_rows) == list(range(63))
