"""
baselines.py - standard comparators for the COCOMO-81 effort models
===================================================================

The first question in effort-estimation research is whether a new model beats
the established ones. This script scores the weighted and unweighted SVR against:

  * Intermediate COCOMO-81 with Boehm's published coefficients (no fitting),
  * COCOMO-81 recalibrated on each training fold (mode intercepts + size exponent),
  * log-linear regression (ridge, penalty chosen by inner CV on training rows only),
  * the training-fold median effort.

Every model sees exactly the same 10 outer folds as train_model.py (shuffled
KFold, random_state=42), and anything fitted is fitted on outer-training rows only.
Accuracy is summarised with absolute-error measures and Standardised Accuracy
(Shepperd & MacDonell, 2012) as well as MMRE, which is biased towards models that
underestimate (Foss et al., 2003). Paired comparisons use per-project absolute
errors: a Wilcoxon signed-rank test and a 95% bootstrap interval for the MAE
difference (projects resampled, models not refitted).

Run:  python baselines.py      (writes baselines.json; about a minute)
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from fahp import compute_fahp_feature_weights
from validation import fit_deployment_model, metrics_from_predictions, nested_cv_evaluate, predict_model

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURE_COLS = [
    "loc", "rely", "data", "cplx", "time", "stor",
    "virt", "turn", "acap", "aexp", "pcap", "vexp",
    "lexp", "modp", "tool", "sced", "dev_mode_ord",
]
MULTIPLIERS = FEATURE_COLS[1:16]          # the 15 COCOMO-81 effort multipliers
DEV_MODE_ORD = {"organic": 1, "semidetached": 2, "embedded": 3}
# Boehm (1981), intermediate COCOMO: effort = a * KLOC^b * EAF (person-months).
INTERMEDIATE = {1: (3.2, 1.05), 2: (3.0, 1.12), 3: (2.8, 1.20)}

# Identical to train_model.py, so the SVR rows reproduce the reported results.
PARAM_GRID = {
    "C": [0.5, 1, 5, 10, 50, 100, 200],
    "gamma": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, "scale"],
    "epsilon": [0.05, 0.1, 0.2],
}
OUTER_SPLITS, INNER_SPLITS, RANDOM_STATE = 10, 5, 42


def load(name):
    df = pd.read_csv(os.path.join(HERE, name))
    df["dev_mode_ord"] = df["dev_mode"].map(DEV_MODE_ORD).astype(float)
    df = df[FEATURE_COLS + ["actual"]].dropna()
    df = df[(df[FEATURE_COLS] > 0).all(axis=1) & (df["actual"] > 0)]
    return df[FEATURE_COLS].values.astype(float), df["actual"].values.astype(float)


def weights(X, y):
    return compute_fahp_feature_weights(X, y, verbose=False)[0]


# ── Comparators: fit(X_train, y_train) -> predict(X) ─────────────────────────

def cocomo_published(X_train=None, y_train=None):
    def predict(X):
        a, b = np.array([INTERMEDIATE[int(m)] for m in X[:, 16]]).T
        return a * X[:, 0] ** b * X[:, 1:16].prod(axis=1)
    return predict


def mode_design(X):
    return np.column_stack([(X[:, 16] == m).astype(float) for m in (1, 2, 3)] + [np.log(X[:, 0])])


def cocomo_calibrated(X_train, y_train):
    """ln(effort / EAF) = ln a_mode + b ln KLOC, least squares on training rows."""
    target = np.log(y_train / X_train[:, 1:16].prod(axis=1))
    coef, *_ = np.linalg.lstsq(mode_design(X_train), target, rcond=None)
    return lambda X: np.exp(mode_design(X) @ coef) * X[:, 1:16].prod(axis=1)


def log_features(X):
    return np.column_stack([np.log(X[:, :16])] + [(X[:, 16] == m).astype(float) for m in (2, 3)])


def log_linear(X_train, y_train):
    """Ridge on log-features and log-effort; penalty by inner 5-fold CV on training rows."""
    inner = KFold(INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=inner).fit(log_features(X_train), np.log(y_train))
    return lambda X: np.exp(model.predict(log_features(X)))


def median_effort(X_train, y_train):
    return lambda X: np.full(len(X), np.median(y_train))


COMPARATORS = {
    "cocomo81_published": ("Intermediate COCOMO-81, published coefficients", cocomo_published),
    "cocomo81_calibrated": ("COCOMO-81 recalibrated per fold", cocomo_calibrated),
    "log_linear_ridge": ("Log-linear regression (ridge, nested)", log_linear),
    "median_effort": ("Training median effort", median_effort),
}


def cross_validated(fit, X, y):
    predictions = np.empty(len(y))
    for train_idx, test_idx in KFold(OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X):
        predictions[test_idx] = fit(X[train_idx], y[train_idx])(X[test_idx])
    return predictions


# ── Measures ────────────────────────────────────────────────────────────────

def random_guess_mae(y):
    """Exact expected MAE of random guessing P0 (Langdon et al., 2016): predict another project's effort."""
    diffs = np.abs(y[:, None] - y[None, :])
    return float(diffs.sum() / (len(y) * (len(y) - 1)))


def summary(y, prediction, mae_p0):
    m = metrics_from_predictions(y, prediction)
    return {"mae": m["mae"], "median_absolute_error": m["median_absolute_error"], "mmre": m["mmre"],
            "mdmre": m["mdmre"], "pred25": m["pred25"], "standardised_accuracy": 1 - m["mae"] / mae_p0}


def paired(y, a, b, resamples=10000, seed=0):
    """Model a vs model b on the same projects. Negative MAE difference: a is better."""
    ea, eb = np.abs(a - y), np.abs(b - y)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(y), (resamples, len(y)))
    boot = ea[pick].mean(axis=1) - eb[pick].mean(axis=1)
    return {"mae_difference": float(ea.mean() - eb.mean()),
            "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
            "wilcoxon_p": float(wilcoxon(ea, eb).pvalue),
            "projects_a_better": int((ea < eb).sum()), "n": len(y)}


def main():
    X, y = load("cocomo81.csv")
    mae_p0 = random_guess_mae(y)
    predictions = {}
    for key, weighted in (("svr_weighted", True), ("svr_unweighted", False)):
        predictions[key] = nested_cv_evaluate(X, y, weights, weighted=weighted, param_grid=PARAM_GRID,
                                              outer_splits=OUTER_SPLITS, inner_splits=INNER_SPLITS,
                                              random_state=RANDOM_STATE)["predictions"]
    for key, (_, fit) in COMPARATORS.items():
        predictions[key] = cross_validated(fit, X, y)

    labels = {"svr_weighted": "SVR, FAHP-proxy weighted (this project)", "svr_unweighted": "SVR, unweighted",
              **{k: label for k, (label, _) in COMPARATORS.items()}}
    results = {"n": len(y), "mae_random_guessing": mae_p0, "labels": labels,
               "cocomo81_cv": {k: summary(y, p, mae_p0) for k, p in predictions.items()},
               "paired_vs_svr_weighted": {k: paired(y, predictions["svr_weighted"], p)
                                          for k, p in predictions.items() if k != "svr_weighted"}}

    # External check: fit on all 63 COCOMO-81 projects, predict NASA93 (never used for selection).
    Xn, yn = load("nasa93.csv")
    bundle, _, _ = fit_deployment_model(X, y, weights, weighted=True, param_grid=PARAM_GRID,
                                        inner_splits=INNER_SPLITS)
    external = {"svr_weighted": predict_model(bundle, Xn)}
    for key, (_, fit) in COMPARATORS.items():
        external[key] = fit(X, y)(Xn)
    mae_p0_n = random_guess_mae(yn)
    results["nasa93_external"] = {"n": len(yn), "mae_random_guessing": mae_p0_n,
                                  **{k: summary(yn, p, mae_p0_n) for k, p in external.items()}}
    results["nasa93_paired_vs_svr_weighted"] = {k: paired(yn, external["svr_weighted"], p)
                                                for k, p in external.items() if k != "svr_weighted"}

    with open(os.path.join(HERE, "baselines.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for split, key in (("COCOMO-81, 10-fold CV", "cocomo81_cv"), ("NASA93, external", "nasa93_external")):
        print(f"\n{split}")
        print(f"{'model':<48}{'MAE':>9}{'MdAE':>8}{'MMRE':>7}{'MdMRE':>7}{'Pred25':>7}{'SA':>7}")
        for k, s in results[key].items():
            if isinstance(s, dict):
                print(f"{labels[k]:<48}{s['mae']:>9.1f}{s['median_absolute_error']:>8.1f}{s['mmre']:>7.3f}"
                      f"{s['mdmre']:>7.3f}{s['pred25']:>7.2f}{s['standardised_accuracy']:>7.2f}")
    print("\nWeighted SVR minus comparator, absolute error (negative = SVR better):")
    for split in ("paired_vs_svr_weighted", "nasa93_paired_vs_svr_weighted"):
        for k, r in results[split].items():
            print(f"  {split.split('_')[0]:<8}{labels[k]:<46}{r['mae_difference']:>+9.1f}  "
                  f"[{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]  p={r['wilcoxon_p']:.3f}  "
                  f"SVR better on {r['projects_a_better']}/{r['n']}")


if __name__ == "__main__":
    main()
