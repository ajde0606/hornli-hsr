import numpy as np
import pandas as pd

# ---------- helpers ----------
def _wls_constrained(X, y, w, C):
    X = np.asarray(X, float); y = np.asarray(y, float)
    w = np.asarray(w, float); C = np.asarray(C, float)
    WX = X * w[:, None]; XtWX = X.T @ WX; XtWy = X.T @ (w * y)
    if C.size == 0:
        return np.linalg.solve(XtWX + 1e-10*np.eye(XtWX.shape[0]), XtWy)
    zero = np.zeros((C.shape[1], C.shape[1]))
    K = np.block([[XtWX, C],[C.T, zero]]) + 1e-10*np.eye(XtWX.shape[0]+C.shape[1])
    rhs = np.concatenate([XtWy, np.zeros(C.shape[1])])
    return np.linalg.solve(K, rhs)[:X.shape[1]]

def _normalize_cap_weights(weights: pd.Series, index: pd.Index) -> pd.Series:
    cw = weights.reindex(index).fillna(0.0).astype(float).clip(lower=0.0)
    s = cw.sum()
    return cw / s if s > 0 else cw

def _block_cap_weight_constraint(block_df: pd.DataFrame, cap_w: pd.Series) -> np.ndarray:
    B = block_df.reindex(cap_w.index).fillna(0.0).astype(float)
    return (B.mul(cap_w, axis=0)).sum(axis=0).values

def _lambda_from_half_life_days(half_life_days: float, *, step_days: float = 1.0) -> float:
    """lambda = 0.5 ** (step_days / half_life_days)"""
    if np.isinf(half_life_days): return 1.0
    if half_life_days <= 0: raise ValueError("half_life_days must be > 0 or inf")
    return float(0.5 ** (step_days / float(half_life_days)))

def _apply_cap_in_reg_weights(base_w: pd.Series,
                              mcap: pd.Series | None,
                              *,
                              exponent: float = 0.5,
                              normalize: str = "mean1") -> pd.Series:
    w = base_w.copy().astype(float)
    if mcap is not None:
        s = mcap.reindex(w.index).astype(float).clip(lower=0.0).fillna(0.0)
        w = w * s.pow(exponent)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if normalize == "mean1":
        m = w.mean()
        if m > 0: w = w / m
    return w


def concat_loadings(X_style, D_ind, C_cty, idx):
    Xs = X_style.loc[idx].fillna(0.0)
    Di = D_ind.loc[idx].fillna(0.0)
    Ct = C_cty.loc[idx].fillna(0.0)

    # Normalize fractional country rows to sum to 1 when present
    row_sums = Ct.sum(axis=1)
    need_norm = (row_sums > 0) & (~np.isclose(row_sums, 1.0))
    if need_norm.any():
        Ct.loc[need_norm] = Ct.loc[need_norm].div(row_sums[need_norm], axis=0)

    X = np.c_[Xs.values, Ct.values, Di.values]
    col_names = list(Xs.columns) + list(Ct.columns) + list(Di.columns)
    return X, col_names


def cross_sectional_regression_one_day(
    r: pd.Series,              # stock returns for the day (index: asset)
    X_style: pd.DataFrame,     # style exposures (index: asset, columns: styles)
    D_ind: pd.DataFrame,       # industry dummies (index: asset, columns: industries)
    C_cty: pd.DataFrame,       # country weights/fractions (index: asset, columns: countries)
    market_cap: pd.Series,     # market caps (index: asset)
    *,
    hsigma: pd.Series | None = None,     # optional prior specific stdevs (index: asset)
    reg_cap_exponent: float = 0.5,
    reg_weight_normalize: str = "mean1",
    return_winsor_p: float = 0.01
) -> tuple[pd.Series, pd.Series]:
    """
    Runs a WLS cross-sectional regression for ONE DAY:
        r = X b + e, with cap-weighted sum-to-zero constraints on countries & industries.
    Returns:
        factor_returns: pd.Series (styles + countries + industries)
        specific_return: pd.Series (per asset)
    """
    idx = r.index
    
    # Winsorize returns
    r_w = r.clip(*r.quantile([return_winsor_p, 1 - return_winsor_p])) if return_winsor_p else r

    # Base weights: inverse prior variance
    if hsigma is None:
        base_w = pd.Series(1.0, index=idx, dtype=float)
    else:
        base_w = (1.0 / (hsigma.reindex(idx).astype(float) ** 2)).replace([np.inf, 0], np.nan).fillna(1.0)

    # √(cap) regression weighting
    w = _apply_cap_in_reg_weights(base_w, market_cap, exponent=reg_cap_exponent, normalize=reg_weight_normalize)

    X, col_names = concat_loadings(Xs, Di, Ct, idx)

    # Cap-weighted sum-to-zero constraints for countries and industries
    C_rows = []
    cap_w = _normalize_cap_weights(market_cap, idx)
    c = _block_cap_weight_constraint(Ct, cap_w)
    if np.linalg.norm(c, 1) > 1e-12:
        row = np.zeros(len(col_names)); s = Xs.shape[1]; row[s:s+Ct.shape[1]] = c; C_rows.append(row)
    c = _block_cap_weight_constraint(Di, cap_w)
    if np.linalg.norm(c, 1) > 1e-12:
        row = np.zeros(len(col_names)); s = Xs.shape[1]+Ct.shape[1]; row[s:s+Di.shape[1]] = c; C_rows.append(row)
    C = np.array(C_rows, float).T if C_rows else np.zeros((len(col_names), 0), float)

    # Solve and residuals
    b = _wls_constrained(X, r_w.values, w.values, C)
    factor_returns = pd.Series(b, index=col_names, name="factor_return")
    fitted = X @ b
    e = pd.Series(r_w.values - fitted, index=idx, name="specific_return")
    return factor_returns, e