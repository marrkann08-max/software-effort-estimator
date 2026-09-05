"""
train_model.py - research prototype inspired by Sehra et al. (2019)
===================================================================

Dataset: COCOMO-81 (63 projects, 1980s defense/aerospace software)
  - Drops rows with NaN / zero feature values
  - Computes pure nested outer-fold signed relative errors → stores 10th/90th pct as
    empirical 80% prediction interval (pi_lo_pct, pi_hi_pct)

Pipeline (mirrored exactly in app.py):
  raw X  →  log(X)  →  MinMaxScaler  →  × sqrt(fahp_w)  →  SVR
  raw y  →  log1p(y) →  MinMaxScaler  →  SVR target

Run:
  python train_model.py

Methodological inspiration: Sehra et al. (2019). This project is an
SVR adaptation, not a reproduction of their weighted-kernel LSSVM solver.
"""

import numpy as np
import pandas as pd
import pickle
import os
from scipy.stats import spearmanr

from explainability import kernel_shap_numerical_sensitivity, repeated_oof_shap_analysis
from metrics_store import update_metrics
from validation import (
    fit_deployment_model,
    metrics_from_predictions,
    nested_cv_evaluate,
    predict_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  DATA
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "loc", "rely", "data", "cplx", "time", "stor",
    "virt", "turn", "acap", "aexp", "pcap", "vexp",
    "lexp", "modp", "tool", "sced", "dev_mode_ord",
]
FEATURE_NAMES = [
    "LOC (KLOC)", "RELY", "DATA", "CPLX", "TIME", "STOR",
    "VIRT", "TURN", "ACAP", "AEXP", "PCAP", "VEXP",
    "LEXP", "MODP", "TOOL", "SCED", "Dev Mode",
]

# ── Load COCOMO-81 ─────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "cocomo81.csv")
df_c81   = pd.read_csv(csv_path)
n_cocomo81_raw = len(df_c81)

# ── Encode development mode (ordinal: organic=1, semidetached=2, embedded=3)
# log(1)=0, log(2)=0.69, log(3)=1.10 — well-spaced after log transform.
# Captures the distinct COCOMO base coefficients (a=2.4/3.0/3.6, b=1.05/1.12/1.20).
DEV_MODE_ORD = {"organic": 1, "semidetached": 2, "embedded": 3}
df_c81["dev_mode_ord"] = df_c81["dev_mode"].map(DEV_MODE_ORD).astype(float)

# ── Clean and prepare ────────────────────────────────────────────────────────
df_all = df_c81[FEATURE_COLS + ["actual", "dev_mode"]].copy()
df_all = df_all.dropna()
df_all = df_all[(df_all[FEATURE_COLS] > 0).all(axis=1)]
df_all = df_all[df_all["actual"] > 0]

n_cocomo81 = len(df_all)
n_nasa93   = 0

X_raw = df_all[FEATURE_COLS].values.astype(float)
y_raw = df_all["actual"].values.astype(float)

print(f"Dataset: {len(df_all)} projects  (COCOMO-81: {n_cocomo81})")
print(f"Effort : min={y_raw.min():.1f}  max={y_raw.max():.1f}  mean={y_raw.mean():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CHANG (1996) EXTENT-ANALYSIS FAHP
# ─────────────────────────────────────────────────────────────────────────────

TFN_SCALE = {
    1: (1.0, 1.0, 1.0),
    2: (1.0, 2.0, 3.0),
    3: (1.0, 3.0, 5.0),
    4: (1.0, 4.0, 7.0),
    5: (3.0, 5.0, 7.0),
    6: (3.0, 6.0, 9.0),
    7: (5.0, 7.0, 9.0),
    8: (7.0, 8.0, 9.0),
    9: (7.0, 9.0, 9.0),
}

def _tfn_reciprocal(tfn):
    l, m, u = tfn
    return (1.0 / u, 1.0 / m, 1.0 / l)

def _continuous_to_tfn(ratio: float):
    ratio = max(1.0 / 9.0, min(9.0, ratio))
    if ratio < 1.0:
        return _tfn_reciprocal(_continuous_to_tfn(1.0 / ratio))
    lo = max(1, min(9, int(np.floor(ratio))))
    hi = max(1, min(9, int(np.ceil(ratio))))
    if lo == hi:
        return TFN_SCALE[lo]
    t = ratio - lo
    return tuple(TFN_SCALE[lo][k] * (1 - t) + TFN_SCALE[hi][k] * t for k in range(3))

def build_pairwise_matrix(values: np.ndarray):
    n   = len(values)
    mat = [[(1.0, 1.0, 1.0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ratio     = float(values[i]) / float(max(values[j], 1e-9))
            tfn       = _continuous_to_tfn(ratio)
            mat[i][j] = tfn
            mat[j][i] = _tfn_reciprocal(tfn)
    return mat

def synthetic_extent_values(mat):
    n        = len(mat)
    row_sums = []
    for i in range(n):
        sl = sum(mat[i][j][0] for j in range(n))
        sm = sum(mat[i][j][1] for j in range(n))
        su = sum(mat[i][j][2] for j in range(n))
        row_sums.append((sl, sm, su))
    tl, tm, tu = [sum(r[k] for r in row_sums) for k in range(3)]
    inv_total  = (1.0 / tu, 1.0 / tm, 1.0 / tl)
    return [(sl * inv_total[0], sm * inv_total[1], su * inv_total[2])
            for (sl, sm, su) in row_sums]

def degree_of_possibility(M1, M2):
    _,  m1, u1 = M1   # l1 not needed by the Chang (1996) formula
    l2, m2, _  = M2   # u2 not needed
    if m1 >= m2:
        return 1.0
    if u1 <= l2:          # M1 entirely below M2 → impossible for M1 ≥ M2
        return 0.0
    denom = (m2 - l2) - (m1 - u1)   # == u1 - m1 + m2 - l2, always > 0 here
    if abs(denom) < 1e-10:
        return 0.0
    return max(0.0, min(1.0, (u1 - l2) / denom))   # correct Chang (1996) formula

def consistency_ratio(mat) -> float:
    n     = len(mat)
    crisp = np.array([[mat[i][j][1] for j in range(n)] for i in range(n)], dtype=float)
    col_sums = crisp.sum(axis=0)
    norm     = crisp / (col_sums + 1e-10)
    weights  = norm.mean(axis=1)
    Aw       = crisp @ weights
    lam_max  = float(np.mean(Aw / (weights + 1e-10)))
    CI       = (lam_max - n) / max(n - 1, 1)
    RI_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI       = RI_table.get(n, 1.54)
    return CI / RI if RI > 0 else 0.0

def fahp_weights_from_values(values: np.ndarray, label="", verbose=True) -> np.ndarray:
    mat = build_pairwise_matrix(values)
    if verbose:
        cr = consistency_ratio(mat)
        if cr > 0.1:
            print(f"  ! {label} CR={cr:.3f} > 0.1 (auto-generated ratios)")
        else:
            print(f"  OK {label} CR={cr:.3f}")
    S       = synthetic_extent_values(mat)
    n       = len(S)
    W_prime = np.array([
        min(degree_of_possibility(S[i], S[j]) for j in range(n) if j != i)
        for i in range(n)
    ])
    total = W_prime.sum()
    return W_prime / total if total > 1e-10 else np.ones(n) / n

def compute_fahp_feature_weights(X_sub, y_sub, verbose=False):
    """FAHP-inspired, data-derived feature weighting for a project subset.

    Chang extent analysis is applied to effort/LOC project comparisons and the
    resulting project weights are mapped to features by absolute Spearman
    association. These are supervised proxy weights, not independent expert
    pairwise judgments.

    Project-level weights via Chang extent analysis on the effort and LOC
    criteria, then feature importance via |Spearman(feature, project_weight)|.
    Returns (fahp_w, W_effort, W_loc, W_project).

    Used both globally (all 63 projects, for the deployment model) and inside
    every inner/outer training partition so held-out outcomes never inform the
    relevant weights.
    """
    W_eff  = fahp_weights_from_values(y_sub,        label="Effort criterion", verbose=verbose)
    W_loc_ = fahp_weights_from_values(X_sub[:, 0],  label="LOC criterion",    verbose=verbose)
    W_proj = 0.5 * W_eff + 0.5 * W_loc_
    W_proj = W_proj / W_proj.sum()
    fcorr  = np.array([
        abs(spearmanr(X_sub[:, fi], W_proj).correlation)
        for fi in range(len(FEATURE_COLS))
    ])
    fcorr  = np.nan_to_num(fcorr, nan=0.0)        # constant column in a fold → 0
    fcorr  = np.clip(fcorr, 0.01, None)
    fw     = fcorr / fcorr.sum()
    return fw, W_eff, W_loc_, W_proj

# ── 1a/1b. Global feature weights (all 63 projects — used for the final model)
print("\n-- Computing FAHP-inspired, data-derived weighting proxy --")
fahp_w, W_effort, W_loc, W_project = compute_fahp_feature_weights(X_raw, y_raw, verbose=True)

print("\nTraining-derived proxy feature weights (top 5):")
for idx in np.argsort(fahp_w)[::-1][:5]:
    print(f"  {FEATURE_NAMES[idx]:>14s}: {fahp_w[idx]:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  NESTED MODEL SELECTION AND PURE OUTER-FOLD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "C": [0.5, 1, 5, 10, 50, 100, 200],
    "gamma": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, "scale"],
    "epsilon": [0.05, 0.1, 0.2],
}
OUTER_SPLITS = 10
INNER_SPLITS = 5
RANDOM_STATE = 42

# Predeclared pilot protocol for the SHAP research extension. Each project is
# explained out-of-fold once per repeat, and every background contains only the
# corresponding outer-training projects.
XAI_REPEATS = 10
XAI_OUTER_SPLITS = 5
XAI_INNER_SPLITS = 3
XAI_NSAMPLES = 256
XAI_TOP_K = 3
XAI_SENSITIVITY_ROWS = 12


def cocomo_feature_weights(X_subset, y_subset):
    return compute_fahp_feature_weights(X_subset, y_subset, verbose=False)[0]


print("\n-- Nested 10x5 CV: proxy-weighted model --")
nested = nested_cv_evaluate(
    X_raw,
    y_raw,
    cocomo_feature_weights,
    weighted=True,
    param_grid=PARAM_GRID,
    outer_splits=OUTER_SPLITS,
    inner_splits=INNER_SPLITS,
    random_state=RANDOM_STATE,
)

print("\n-- Nested 10x5 CV: independently tuned unweighted ablation --")
ablation = nested_cv_evaluate(
    X_raw,
    y_raw,
    cocomo_feature_weights,
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
fahp_effect_pct = (ablation_mmre - nested_mmre) / ablation_mmre * 100

print(
    f"Nested CV MMRE={nested_mmre:.4f}; PRED(25)={pred25*100:.1f}%; "
    f"PRED(50)={pred50*100:.1f}%"
)
print(
    f"Weighting comparison: unweighted={ablation_mmre:.4f}, weighted={nested_mmre:.4f}, "
    f"change={fahp_effect_pct:+.1f}%"
)

print("\n-- Repeated nested out-of-fold KernelSHAP research analysis --")
explainability, oof_explanation_records = repeated_oof_shap_analysis(
    X_raw,
    y_raw,
    cocomo_feature_weights,
    PARAM_GRID,
    FEATURE_NAMES,
    repeats=XAI_REPEATS,
    outer_splits=XAI_OUTER_SPLITS,
    inner_splits=XAI_INNER_SPLITS,
    background_size=None,
    nsamples=XAI_NSAMPLES,
    top_k=XAI_TOP_K,
    random_state=RANDOM_STATE + 400,
)
oof_explanations_path = os.path.join(
    os.path.dirname(__file__), "explainability_oof.csv"
)
pd.DataFrame(oof_explanation_records).to_csv(oof_explanations_path, index=False)

# Per-mode metrics use only pure outer-fold predictions.
dev_mode_col = df_all["dev_mode"].values
mode_mmre_results = {}
for mode_key, mode_label in [
    ("organic", "Organic"),
    ("semidetached", "Semi-detached"),
    ("embedded", "Embedded"),
]:
    mask = dev_mode_col == mode_key
    if mask.sum():
        mode_metrics = metrics_from_predictions(y_raw[mask], nested["predictions"][mask])
        mode_mmre_results[mode_key] = {
            "mmre": round(mode_metrics["mmre"], 4),
            "n": int(mask.sum()),
            "pred25": round(mode_metrics["pred25"], 4),
        }
        print(
            f"  {mode_label:>14s}: n={mask.sum():>2d}  "
            f"MMRE={mode_metrics['mmre']:.4f}  PRED(25)={mode_metrics['pred25']*100:.1f}%"
        )

# The deployed model uses parameters selected on all COCOMO rows.  Its training
# fit is never reported as a validation score.
print("\n-- Selecting deployment hyperparameters on all COCOMO-81 rows --")
final_bundle, deployment_params, deployment_selection_mmre = fit_deployment_model(
    X_raw,
    y_raw,
    cocomo_feature_weights,
    weighted=True,
    param_grid=PARAM_GRID,
    inner_splits=INNER_SPLITS,
    random_state=RANDOM_STATE + 100,
)
final_model = final_bundle.model
scaler_X = final_bundle.scaler_X
scaler_y = final_bundle.scaler_y
fahp_w = final_bundle.feature_weights
kernel_w = final_bundle.kernel_weights
X_w = scaler_X.transform(np.log(np.clip(X_raw, 1e-3, None))) * kernel_w

train_pred_pm = predict_model(final_bundle, X_raw)
train_mmre = float(np.mean(np.abs(y_raw - train_pred_pm) / y_raw))
train_rmse = float(np.sqrt(np.mean((y_raw - train_pred_pm) ** 2)))

# Hold the deployed model, background, and evaluation rows fixed to isolate
# KernelSHAP seed/sample-size variation from the joint repeated-CV stability.
sensitivity_indices = np.unique(
    np.linspace(0, len(X_raw) - 1, XAI_SENSITIVITY_ROWS, dtype=int)
)
explainability["numerical_sensitivity"] = kernel_shap_numerical_sensitivity(
    final_bundle,
    X_raw[sensitivity_indices],
    X_raw,
    main_nsamples=XAI_NSAMPLES,
    top_k=XAI_TOP_K,
    random_state=RANDOM_STATE + 600,
    sample_sizes=(128, 256, 512),
    reference_nsamples=1024,
    seed_count=3,
)
explainability["numerical_sensitivity"]["evaluation_row_indices"] = [
    int(index) for index in sensitivity_indices
]
explainability["weighting_proxy"]["full_data_diagnostics"] = {
    "project_count": int(len(X_raw)),
    "effort_project_weights_nonzero": int(np.count_nonzero(W_effort > 1e-12)),
    "loc_project_weights_nonzero": int(np.count_nonzero(W_loc > 1e-12)),
    "combined_project_weights_nonzero": int(np.count_nonzero(W_project > 1e-12)),
    "zero_threshold": "1e-12",
    "effort_pairwise_consistency_ratio": round(
        consistency_ratio(build_pairwise_matrix(y_raw)), 6
    ),
    "loc_pairwise_consistency_ratio": round(
        consistency_ratio(build_pairwise_matrix(X_raw[:, 0])), 6
    ),
    "caveat": "Chang extent analysis produces sparse project weights here; proxy-weight sensitivity and genuine expert elicitation remain required for a confirmatory study.",
}

# Cross-dataset validation is pure: NASA-93 is never used for model selection.
nasa93_mmre = None
nasa93_pred25 = None
n_nasa93_valid = 0
nasa93_path = os.path.join(os.path.dirname(__file__), "nasa93.csv")
if os.path.exists(nasa93_path):
    df_n93 = pd.read_csv(nasa93_path)
    df_n93["dev_mode_ord"] = df_n93["dev_mode"].map(DEV_MODE_ORD).astype(float)
    df_n93_clean = df_n93[FEATURE_COLS + ["actual"]].dropna()
    df_n93_clean = df_n93_clean[(df_n93_clean[FEATURE_COLS] > 0).all(axis=1)]
    df_n93_clean = df_n93_clean[df_n93_clean["actual"] > 0]
    n_nasa93_valid = len(df_n93_clean)
    if n_nasa93_valid:
        X_n93_raw = df_n93_clean[FEATURE_COLS].values.astype(float)
        y_n93_raw = df_n93_clean["actual"].values.astype(float)
        p_n93_pm = predict_model(final_bundle, X_n93_raw)
        nasa_metrics = metrics_from_predictions(y_n93_raw, p_n93_pm)
        nasa93_mmre = nasa_metrics["mmre"]
        nasa93_pred25 = nasa_metrics["pred25"]

best_C = deployment_params["C"]
best_gamma = deployment_params["gamma"]
best_epsilon = deployment_params["epsilon"]

artifacts = {
    "model": final_model,
    "scaler_X": scaler_X,
    "scaler_y": scaler_y,
    "fahp_weights": fahp_w,
    "kernel_w": kernel_w,
    "X_train": X_w,
    "y_train": y_raw,
    "feature_names": FEATURE_NAMES,
    "log_transform": True,
    "best_C": float(best_C),
    "best_gamma": best_gamma,
    "best_epsilon": float(best_epsilon),
    "nested_mmre": round(nested_mmre, 4),
    "loo_mmre": round(nested_mmre, 4),  # compatibility alias; UI labels nested CV
    "validation_method": "nested_10x5_kfold",
    "outer_fold_params": nested["fold_details"],
    "train_mmre": round(train_mmre, 4),
    "train_rmse": round(train_rmse, 2),
    "n_train": len(y_raw),
    "n_cocomo81": n_cocomo81,
    "n_nasa93": n_nasa93_valid,
    "importance": fahp_w,
    "pi_q10": pi_q10,
    "pi_q90": pi_q90,
    "pi_lo_pct": pi_q10,  # compatibility alias for the stored residual quantile
    "pi_hi_pct": pi_q90,
    "residual_definition": "(prediction - actual) / actual",
    "pred25": round(pred25, 4),
    "pred50": round(pred50, 4),
    "mdmre": round(nested["mdmre"], 4),
    "mae": round(nested["mae"], 4),
    "median_absolute_error": round(nested["median_absolute_error"], 4),
    "rmse": round(nested["rmse"], 4),
    "ablation_mmre": round(ablation_mmre, 4),
    "fahp_effect_pct": round(fahp_effect_pct, 1),
    "fahp_improve_pct": round(fahp_effect_pct, 1),
    "mode_mmre": mode_mmre_results,
    "nasa93_mmre": round(nasa93_mmre, 4) if nasa93_mmre is not None else None,
    "nasa93_pred25": round(nasa93_pred25, 4) if nasa93_pred25 is not None else None,
    "n_nasa93_valid": n_nasa93_valid,
    "project_weights_effort": W_effort,
    "project_weights_loc": W_loc,
    "project_weights": W_project,
    "X_raw": X_raw,
    "y_raw_pm": y_raw,
    "explainability": explainability,
}

out_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(out_path, "wb") as stream:
    pickle.dump(artifacts, stream)

methodology = {
    "validation": "Nested 10-fold outer / 5-fold inner shuffled KFold",
    "outer_role": "Pure evaluation only; hyperparameters are never selected on outer-test rows",
    "inner_role": "Hyperparameter selection and preprocessing fit on outer-training rows only",
    "residual_definition": "(prediction - actual) / actual",
    "prediction_interval_formula": "[prediction / (1 + q90), prediction / (1 + q10)]",
}
update_metrics(
    "cocomo81",
    {
        "label": "COCOMO-81",
        "n": n_cocomo81,
        "nested_cv": {
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "mmre": round(nested_mmre, 4),
            "pred25": round(pred25, 4),
            "pred50": round(pred50, 4),
            "mdmre": round(nested["mdmre"], 4),
            "mae_person_months": round(nested["mae"], 4),
            "median_absolute_error_person_months": round(
                nested["median_absolute_error"], 4
            ),
            "rmse_person_months": round(nested["rmse"], 4),
        },
        "prediction_interval": {
            "coverage": 0.80,
            "q10": round(pi_q10, 6),
            "q90": round(pi_q90, 6),
            "source": "pure nested outer-fold signed relative errors",
        },
        "fahp_comparison": {
            "weighted_mmre": round(nested_mmre, 4),
            "unweighted_mmre": round(ablation_mmre, 4),
            "effect_pct": round(fahp_effect_pct, 1),
            "weighted_mdmre": round(nested["mdmre"], 4),
            "unweighted_mdmre": round(ablation["mdmre"], 4),
            "weighted_mae_person_months": round(nested["mae"], 4),
            "unweighted_mae_person_months": round(ablation["mae"], 4),
        },
        "external_nasa93": {
            "n": n_nasa93_valid,
            "mmre": round(nasa93_mmre, 4) if nasa93_mmre is not None else None,
            "pred25": round(nasa93_pred25, 4) if nasa93_pred25 is not None else None,
            "mdmre": round(nasa_metrics["mdmre"], 4) if nasa93_mmre is not None else None,
            "mae_person_months": round(nasa_metrics["mae"], 4) if nasa93_mmre is not None else None,
            "role": "external test; never used for selection",
        },
        "deployment_params": deployment_params,
        "deployment_inner_cv_mmre": round(deployment_selection_mmre, 4),
        "mode_metrics": mode_mmre_results,
        "explainability": explainability,
    },
    methodology=methodology,
)

print(f"\nmodel.pkl saved: {out_path}")
print(f"Nested CV MMRE : {nested_mmre:.4f}")
print(f"PRED(25)       : {pred25*100:.1f}%")
print(f"Unweighted MMRE: {ablation_mmre:.4f}")
print(f"Weighting effect: {fahp_effect_pct:+.1f}%")
if nasa93_mmre is not None:
    print(f"NASA-93 MMRE   : {nasa93_mmre:.4f} (pure external test, n={n_nasa93_valid})")
print(f"Residual q10/q90: {pi_q10:+.4f}, {pi_q90:+.4f}")
print(f"Deployment HPs : {deployment_params}")
print(
    "Primary weighting-proxy/unweighted-SHAP rho: "
    f"{explainability['primary_alignment']['spearman_rho']:+.3f}"
)
rho_interval = explainability["primary_alignment_project_bootstrap"]["metrics"][
    "spearman_rho"
]
print(
    "Conditional project-resampling 95% rho interval: "
    f"[{rho_interval['lower']:+.3f}, {rho_interval['upper']:+.3f}]"
)
print(
    "Unweighted SHAP repeat stability rho: "
    f"{explainability['unweighted_global_ranking_stability']['spearman_rho']['median']:+.3f} median"
)
repeated_accuracy = explainability["repeated_accuracy_descriptive"]
print(
    "Proxy weighting lowered repeated-CV MMRE in "
    f"{repeated_accuracy['weighted_better_repeat_count']}/{repeated_accuracy['repeat_count']} repeats"
)
print(f"OOF explanation records: {oof_explanations_path}")
