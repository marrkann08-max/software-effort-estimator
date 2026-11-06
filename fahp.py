"""Chang (1996) extent-analysis FAHP and the training-derived feature-weighting proxy.

Shared by train_model.py (the reported results and deployed model) and
baselines.py (standard comparators on the same folds). Moved here unchanged
from train_model.py so the weights can be computed without retraining everything.
"""

import numpy as np
from scipy.stats import spearmanr


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
        for fi in range(X_sub.shape[1])
    ])
    fcorr  = np.nan_to_num(fcorr, nan=0.0)        # constant column in a fold → 0
    fcorr  = np.clip(fcorr, 0.01, None)
    fw     = fcorr / fcorr.sum()
    return fw, W_eff, W_loc_, W_proj
