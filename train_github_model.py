"""
train_github_model.py  —  supervised association weighting + SVR on GitHub metrics
This is a complementary modern-data experiment, not a direct reproduction of
Sehra et al. (2019). Its weights are learned from the effort target rather than
elicited through expert FAHP pairwise comparisons.

Features: contributor_count, avg_pr_review_days, pr_merge_rate,
          test_file_ratio, lang_count, commit_frequency
Target:   author_months (derived effort proxy: unique author×date pairs / 20)

Run: python train_github_model.py
Saves: github_model.pkl
"""

import numpy as np
import pandas as pd
import pickle, os, warnings
from scipy.stats import spearmanr

from metrics_store import update_metrics
from validation import (
    fit_deployment_model,
    holdout_evaluate,
    nested_cv_evaluate,
    predict_model,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  DATA
# ─────────────────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "github_projects.csv")
df = pd.read_csv(csv_path)

# Features kept after data quality audit:
#   contributor_count  — r=+0.69, real variance (36–369)
#   avg_pr_review_days — real variance (0.01–6.4d)
#   pr_merge_rate      — real variance (0.20–0.88)
#   test_file_ratio    — now properly computed from git tree API
#   lang_count         — real variance (1–14)
#   commit_frequency   — real variance
# Dropped:
#   has_ci      — zero variance (all modern repos have CI)
#   total_kloc  — disk-size proxy, r=-0.058 with effort
#   stars       — popularity ≠ effort
FEATURE_COLS = [
    "contributor_count",
    "avg_pr_review_days",
    "pr_merge_rate",
    "test_file_ratio",
    "lang_count",
    "commit_frequency",
    "contributors_x_freq",   # interaction: large fast-moving teams
    "ai_tools_count",        # AI coding tool adoption (Copilot, Cursor, Claude…)
]
FEATURE_NAMES = [
    "Contributors",
    "PR Review Time (days)",
    "PR Merge Rate",
    "Test Coverage Ratio",
    "Language Count",
    "Commit Frequency",
    "Contributors x Frequency",
    "AI Tool Adoption",
]
TARGET = "author_months"

df_work = df.copy()
df_work["contributors_x_freq"] = df_work["contributor_count"] * df_work["commit_frequency"]
# ai_tools_count: 0 if column missing (older CSV without patch)
if "ai_tools_count" not in df_work.columns:
    df_work["ai_tools_count"] = 0
else:
    df_work["ai_tools_count"] = df_work["ai_tools_count"].fillna(0)

base_cols = ["contributor_count","avg_pr_review_days","pr_merge_rate",
             "test_file_ratio","lang_count","commit_frequency",
             "contributors_x_freq","ai_tools_count"]
df_clean = df_work[base_cols + [TARGET, "repo", "stars"]].dropna()
df_clean = df_clean[(df_clean[["contributor_count","avg_pr_review_days","pr_merge_rate",
                                "test_file_ratio","lang_count","commit_frequency"]] > 0).all(axis=1)]
df_clean = df_clean[df_clean[TARGET] > 0]

X_raw  = df_clean[FEATURE_COLS].values.astype(float)
y_raw  = df_clean[TARGET].values.astype(float)
repos  = df_clean["repo"].values

n = len(df_clean)
print(f"GitHub dataset: {n} projects")
print(f"Effort: min={y_raw.min():.1f}  max={y_raw.max():.1f}  "
      f"mean={y_raw.mean():.1f}  std={y_raw.std():.1f} PM")
print(f"\nFeature correlations with effort:")
for i, name in enumerate(FEATURE_NAMES):
    corr = float(spearmanr(X_raw[:, i], y_raw).correlation)
    print(f"  {name:>25s}: r={corr:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SUPERVISED DATA-DERIVED FEATURE WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────
# Feature weights are normalized absolute Spearman correlations with effort.
# This is supervised association weighting, not FAHP or expert elicitation.
def compute_gh_feature_weights(X_sub, y_sub):
    """Normalized |Spearman(feature, effort)| feature weights from a subset.

    Because this uses the effort target directly, it is recomputed inside every
    inner and outer training split. Pure outer-test rows never inform weights.
    """
    fc = np.array([
        abs(spearmanr(X_sub[:, i], y_sub).correlation)
        for i in range(len(FEATURE_COLS))
    ])
    fc = np.nan_to_num(fc, nan=0.0)        # constant column in a fold → 0
    fc = np.clip(fc, 0.01, None)
    return fc / fc.sum()

print("\n-- Supervised feature weights (normalized absolute Spearman correlation) --")
supervised_w = compute_gh_feature_weights(X_raw, y_raw)

print("Supervised association weights:")
for i in np.argsort(supervised_w)[::-1]:
    print(f"  {FEATURE_NAMES[i]:>25s}: {supervised_w[i]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  NESTED MODEL SELECTION AND PURE OUTER-FOLD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "C": [0.1, 0.5, 1, 5, 10, 50, 100],
    "gamma": [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, "scale"],
    "epsilon": [0.05, 0.1, 0.2],
}
OUTER_SPLITS = 10
INNER_SPLITS = 5
RANDOM_STATE = 42
COMMIT_SAMPLE_LIMIT = 1200
EFFORT_CAP_AUTHOR_MONTHS = COMMIT_SAMPLE_LIMIT / 20

print("\n-- Nested 10x5 CV: supervised Spearman-weighted GitHub model --")
nested = nested_cv_evaluate(
    X_raw,
    y_raw,
    compute_gh_feature_weights,
    weighted=True,
    param_grid=PARAM_GRID,
    outer_splits=OUTER_SPLITS,
    inner_splits=INNER_SPLITS,
    random_state=RANDOM_STATE,
)

print("\n-- Nested 10x5 CV: independently tuned unweighted GitHub model --")
ablation = nested_cv_evaluate(
    X_raw,
    y_raw,
    compute_gh_feature_weights,
    weighted=False,
    param_grid=PARAM_GRID,
    outer_splits=OUTER_SPLITS,
    inner_splits=INNER_SPLITS,
    random_state=RANDOM_STATE,
)

nested_mmre = nested["mmre"]
pred25 = nested["pred25"]
pred50 = nested["pred50"]
pi_q10 = nested["pi_q10"]
pi_q90 = nested["pi_q90"]
ablation_mmre = ablation["mmre"]
weighting_effect_pct = (ablation_mmre - nested_mmre) / ablation_mmre * 100
naive_mmre = float(np.mean(np.abs(y_raw - y_raw.mean()) / y_raw))

print(
    f"Nested CV MMRE={nested_mmre:.4f}; PRED(25)={pred25*100:.1f}%; "
    f"PRED(50)={pred50*100:.1f}%"
)
print(
    f"Supervised-weighting comparison: unweighted={ablation_mmre:.4f}, "
    f"weighted={nested_mmre:.4f}, change={weighting_effect_pct:+.1f}%"
)

# Actual creation/commit dates were not retained in the current CSV.  Therefore
# the former "temporal" claim is removed.  This is a popularity-based stress
# test only.  Parameter selection is confined to the high-star training split.
print("\n-- Hold-out by repository popularity (not temporal), 70/30 --")
df_sorted = df_clean.sort_values("stars", ascending=False).reset_index(drop=True)
split_n = int(len(df_sorted) * 0.70)
df_tr = df_sorted.iloc[:split_n]
df_te = df_sorted.iloc[split_n:]
popularity = holdout_evaluate(
    df_tr[FEATURE_COLS].values.astype(float),
    df_tr[TARGET].values.astype(float),
    df_te[FEATURE_COLS].values.astype(float),
    df_te[TARGET].values.astype(float),
    compute_gh_feature_weights,
    weighted=True,
    param_grid=PARAM_GRID,
    inner_splits=INNER_SPLITS,
    random_state=RANDOM_STATE + 200,
)
print(
    f"Popularity hold-out MMRE={popularity['mmre']:.4f}; "
    f"PRED(25)={popularity['pred25']*100:.1f}%"
)

print("\n-- Selecting deployment hyperparameters on all GitHub rows --")
final_bundle, deployment_params, deployment_selection_mmre = fit_deployment_model(
    X_raw,
    y_raw,
    compute_gh_feature_weights,
    weighted=True,
    param_grid=PARAM_GRID,
    inner_splits=INNER_SPLITS,
    random_state=RANDOM_STATE + 100,
)
final_model = final_bundle.model
scaler_X = final_bundle.scaler_X
scaler_y = final_bundle.scaler_y
supervised_w = final_bundle.feature_weights
kernel_w = final_bundle.kernel_weights
train_pred = predict_model(final_bundle, X_raw)
train_mmre = float(np.mean(np.abs(y_raw - train_pred) / y_raw))

best_C = deployment_params["C"]
best_g = deployment_params["gamma"]
best_e = deployment_params["epsilon"]
artifacts = {
    "model": final_model,
    "scaler_X": scaler_X,
    "scaler_y": scaler_y,
    "fahp_weights": supervised_w,  # Legacy compatibility key.
    "feature_weighting_method": "supervised_normalized_absolute_spearman",
    "expert_derived_weights": False,
    "kernel_w": kernel_w,
    "X_raw": X_raw,
    "y_raw": y_raw,
    "repos": repos,
    "feature_names": FEATURE_NAMES,
    "feature_cols": FEATURE_COLS,
    "nested_mmre": round(nested_mmre, 4),
    "loo_mmre": round(nested_mmre, 4),  # compatibility alias; UI labels nested CV
    "validation_method": "nested_10x5_kfold",
    "outer_fold_params": nested["fold_details"],
    "train_mmre": round(train_mmre, 4),
    "pred25": round(pred25, 4),
    "pred50": round(pred50, 4),
    "pi_q10": pi_q10,
    "pi_q90": pi_q90,
    "pi_lo_pct": pi_q10,
    "pi_hi_pct": pi_q90,
    "residual_definition": "(prediction - actual) / actual",
    "ablation_mmre": round(ablation_mmre, 4),
    "fahp_effect_pct": round(weighting_effect_pct, 1),  # Legacy compatibility key.
    "fahp_improve_pct": round(weighting_effect_pct, 1),  # Legacy compatibility key.
    "naive_mmre": round(naive_mmre, 4),
    "popularity_holdout_mmre": round(popularity["mmre"], 4),
    "popularity_holdout_pred25": round(popularity["pred25"], 4),
    "popularity_train_n": split_n,
    "popularity_test_n": len(df_te),
    "popularity_params": popularity["params"],
    "commit_sample_limit": COMMIT_SAMPLE_LIMIT,
    "effort_cap_author_months": EFFORT_CAP_AUTHOR_MONTHS,
    "best_C": float(best_C),
    "best_gamma": best_g,
    "best_epsilon": float(best_e),
    "n_projects": n,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "github_model.pkl")
with open(out, "wb") as stream:
    pickle.dump(artifacts, stream)

update_metrics(
    "github",
    {
        "label": "GitHub open-source proof of concept",
        "n": n,
        "nested_cv": {
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "mmre": round(nested_mmre, 4),
            "pred25": round(pred25, 4),
            "pred50": round(pred50, 4),
        },
        "prediction_interval": {
            "coverage": 0.80,
            "q10": round(pi_q10, 6),
            "q90": round(pi_q90, 6),
            "source": "pure nested outer-fold signed relative errors",
        },
        "fahp_comparison": {  # Legacy compatibility key.
            "weighted_mmre": round(nested_mmre, 4),
            "unweighted_mmre": round(ablation_mmre, 4),
            "effect_pct": round(weighting_effect_pct, 1),
            "method": "supervised_normalized_absolute_spearman",
            "expert_derived": False,
        },
        "popularity_holdout": {
            "train_n": split_n,
            "test_n": len(df_te),
            "mmre": round(popularity["mmre"], 4),
            "pred25": round(popularity["pred25"], 4),
            "selection": "inner 5-fold CV on high-star training partition only",
            "caveat": "Popularity hold-out; repository stars are not a time variable",
        },
        "effort_proxy": {
            "definition": "unique author-days in sampled commits divided by 20",
            "commit_sample_limit": COMMIT_SAMPLE_LIMIT,
            "approximate_cap_author_months": EFFORT_CAP_AUTHOR_MONTHS,
            "validated_above_cap": False,
        },
        "deployment_params": deployment_params,
        "deployment_inner_cv_mmre": round(deployment_selection_mmre, 4),
    },
)

print(f"\ngithub_model.pkl saved: {out}")
print(f"Nested CV MMRE : {nested_mmre:.4f}")
print(f"PRED(25)       : {pred25*100:.1f}%")
print(f"Unweighted MMRE: {ablation_mmre:.4f}")
print(f"Weighting effect: {weighting_effect_pct:+.1f}%")
print(f"Popularity hold-out MMRE: {popularity['mmre']:.4f}")
print(f"Effort proxy cap: approximately {EFFORT_CAP_AUTHOR_MONTHS:.0f} author-months")
print(f"Deployment HPs: {deployment_params}")
