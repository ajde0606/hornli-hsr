

import numpy as np
import pandas as pd

def _weighted_mean(x, w):
    w = w.reindex(x.index).astype(float)
    return (w * x).sum() / max(w.sum(), 1e-12)

def _weighted_std(x, w):
    m = _weighted_mean(x, w)
    var = (w * (x - m) ** 2).sum() / max(w.sum(), 1e-12)
    return np.sqrt(max(var, 1e-12))

def _standardize_styles(Xs, w):
    out = {}
    for c in Xs.columns:
        mu = _weighted_mean(Xs[c], w)
        sd = _weighted_std(Xs[c], w)
        out[c] = (Xs[c] - mu) / (sd if sd > 0 else 1.0)
    return pd.DataFrame(out, index=Xs.index)

def _wls_constrained(X, y, w, C):  # C' b = 0 (one or more linear constraints)
    WX = X * w[:, None]
    XtWX = X.T @ WX
    XtWy = X.T @ (w * y)
    zero = np.zeros((C.shape[1], C.shape[1]))
    K = np.block([[XtWX, C], [C.T, zero]])
    rhs = np.concatenate([XtWy, np.zeros(C.shape[1])])
    K += 1e-10 * np.eye(K.shape[0])  # tiny ridge
    sol = np.linalg.solve(K, rhs)
    return sol[:X.shape[1]]

def _block_sum_to_zero_constraint(block_df, weights):
    """Return a constraint vector c s.t. sum_i w_i * block_ij * beta_j = 0  (one constraint for the whole block)."""
    w = weights.reindex(block_df.index).fillna(0.0).astype(float)
    # Column sums under weights
    col_wsum = (block_df.mul(w, axis=0)).sum(axis=0).values  # shape (K,)
    return col_wsum  # 1 x K

def cross_sectional_risk_model_with_country(
    r: pd.Series,
    X_style: pd.DataFrame,
    D_ind: pd.DataFrame,
    C_cty: pd.DataFrame,  # NEW: country exposures (one-hot or revenue weights)
    *,
    hsigma: pd.Series | None = None,
    style_winsor_p: float = 0.01,
    return_winsor_p: float = 0.01,
    ewma_lambda: float = 0.94,
    prev_spec_var: pd.Series | None = None,
    # Which weights to use for the sum-to-zero constraints (market-cap is typical)
    constraint_weights: pd.Series | None = None,
    # If constraint_weights is None, fall back to WLS weights for constraints
    constrain_industries: bool = True,
    constrain_countries: bool = True,
):
    """
    Regress r on [standardized styles | countries | industries] with WLS and
    sum-to-zero constraints on country and industry coefficients.

    Inputs must share the same index of assets for this cross-section.
    """
    idx = r.index
    Xs = X_style.reindex(idx)
    Di = D_ind.reindex(idx)
    Ct = C_cty.reindex(idx)

    # Normalize country rows if they are fractional but not guaranteed to sum to 1
    row_sums = Ct.sum(axis=1)
    need_norm = (row_sums > 0) & (~np.isclose(row_sums, 1.0))
    if need_norm.any():
        Ct.loc[need_norm] = Ct.loc[need_norm].div(row_sums[need_norm], axis=0)

    # Winsorize returns lightly
    if return_winsor_p:
        lo, hi = r.quantile([return_winsor_p, 1 - return_winsor_p])
        r_w = r.clip(lo, hi)
    else:
        r_w = r

    # WLS weights from prior specific risk
    if hsigma is None:
        w = pd.Series(1.0, index=idx)
    else:
        w = 1.0 / (hsigma.reindex(idx).astype(float) ** 2)
        w = w.replace([np.inf, 0], np.nan).fillna(1.0)

    # Standardize styles cross-sectionally
    Xs_w = Xs.copy()
    if style_winsor_p:
        for c in Xs.columns:
            lo, hi = Xs[c].quantile([style_winsor_p, 1 - style_winsor_p])
            Xs_w[c] = Xs[c].clip(lo, hi)
    Xs_z = _standardize_styles(Xs_w, w)

    # Design matrix: [styles | countries | industries]
    X_blocks = [Xs_z.values, Ct.values, Di.values]
    X = np.c_[*X_blocks]
    col_names = list(Xs_z.columns) + list(Ct.columns) + list(Di.columns)

    # Build constraints
    cw = constraint_weights.reindex(idx) if constraint_weights is not None else w

    C_list = []
    start = 0
    # styles block -> no constraint
    start += Xs_z.shape[1]

    if constrain_countries:
        c_cty = np.zeros(len(col_names))
        c_cty[start : start + Ct.shape[1]] = _block_sum_to_zero_constraint(Ct, cw)
        C_list.append(c_cty)
    start += Ct.shape[1]

    if constrain_industries:
        c_ind = np.zeros(len(col_names))
        c_ind[start : start + Di.shape[1]] = _block_sum_to_zero_constraint(Di, cw)
        C_list.append(c_ind)

    C = np.stack(C_list, axis=1) if C_list else np.zeros((len(col_names), 0))

    # Solve constrained WLS
    b = _wls_constrained(X, r_w.values, w.values, C) if C.shape[1] > 0 else _wls_constrained(X, r_w.values, w.values, np.zeros((X.shape[1], 0)))
    factor_returns = pd.Series(b, index=col_names)

    # Specific returns and variance update
    fitted = X @ b
    e = pd.Series(r_w.values - fitted, index=idx, name="specific_return")
    e2 = e.pow(2).clip(lower=1e-12)

    if prev_spec_var is None:
        spec_var = e2
    else:
        pv = prev_spec_var.reindex(idx).fillna(e2)
        spec_var = ewma_lambda * pv + (1.0 - ewma_lambda) * e2

    return factor_returns, e.rename("specific_return"), spec_var.rename("specific_variance")