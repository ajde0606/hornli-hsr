import numpy as np
import pandas as pd
from hsr.config import *

styles = [
    "beta",
    "volatility",
    "momentum",
    "size",
    "nonlinear_size",
    "trading_activity",
    "growth",
    "earnings_yield",
    "value",
    "earnings_variability",
    "leverage",
    "dividend_yield"
]


def _rowwise_cap_weights(caps_like_X: pd.DataFrame, mask_like_X: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize cap weights per date over valid entries of X.
    Negative/NaN caps -> 0; rows with no valid weights -> NaN (we'll handle later).
    """
    w = caps_like_X.where(mask_like_X).astype(float).clip(lower=0.0)
    row_sum = w.sum(axis=1)
    w = w.div(row_sum.replace(0.0, np.nan), axis=0)
    return w

def _rowwise_weighted_mean_std(X: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Cap-weighted row mean and std (population, ddof=0)."""
    # mean
    mu_w = (X * w).sum(axis=1)
    # variance
    xc = X.sub(mu_w, axis=0)
    var_w = (w * xc * xc).sum(axis=1)
    sigma_w = var_w.pow(0.5)
    return mu_w, sigma_w

def winsorize_and_standardize_descriptor(
    df_desc: pd.DataFrame,          # index=date, columns=tickers
    name: str,
    cap_df: pd.DataFrame,           # same shape (date × ticker) market caps (or ADV); nonnegatives
    *,
    p_low: float = 0.001,           # first-pass EW quantile clip
    p_high: float = 0.999,
    trim_sigma: float = 3.0,        # sigma clip around cap-weighted mean/std
    min_valid_per_row: int = 25     # skip rows with too few names
) -> pd.DataFrame:
    """
    Returns a DataFrame Z (date × ticker) with:
      - cap-weighted mean ≈ 0 each date
      - cap-weighted stdev ≈ 1 each date
      - robust two-stage winsorization (EW-quantile then cap-weighted sigma clip)
    """
    print("winsorizing and standardizing", name)

    # 0) Align and clean
    X = df_desc.astype(float).copy()
    caps = cap_df.reindex_like(X).astype(float)
    mask = X.notna() & caps.notna()

    # Optional: rows with too few valid observations -> all NaN (avoid unstable stats)
    valid_counts = mask.sum(axis=1)
    X.loc[valid_counts < min_valid_per_row, :] = np.nan
    mask = X.notna() & caps.notna()

    # 1) First-pass EW tail clip by row (robustness against extreme outliers)
    if p_low is not None and p_high is not None:
        q_low = X.quantile(p_low, axis=1, interpolation="linear")
        q_high = X.quantile(p_high, axis=1, interpolation="linear")
        X = X.clip(lower=q_low, upper=q_high, axis=0)

    # 2) Build cap weights (normalized per date over valid X)
    w = _rowwise_cap_weights(caps_like_X=caps, mask_like_X=X.notna())

    # 3) Cap-weighted sigma clip around cap-weighted mean/std
    mu_w, sigma_w = _rowwise_weighted_mean_std(X, w)
    # Guard tiny/zero sigma (avoid division by 0)
    sigma_w = sigma_w.replace(0.0, np.nan)

    lower = mu_w - trim_sigma * sigma_w
    upper = mu_w + trim_sigma * sigma_w
    X = X.clip(lower=lower, upper=upper, axis=0)

    # 4) Recompute weights after clipping (mask may change if NaNs were created)
    w = _rowwise_cap_weights(caps_like_X=caps, mask_like_X=X.notna())

    # 5) Cap-weighted standardization to mean 0, stdev 1
    mu_w, sigma_w = _rowwise_weighted_mean_std(X, w)
    sigma_w = sigma_w.replace(0.0, np.nan)

    Xc = X.sub(mu_w, axis=0)
    Z = Xc.div(sigma_w, axis=0)

    # 6) Leave rows with insufficient support as NaN
    Z.loc[valid_counts < min_valid_per_row, :] = np.nan

    # Optional: tiny numerical cleanup
    Z = Z.astype(float)

    Z.columns.name = identifier
    Z.index.name = date_col
    Z = Z.unstack()
    Z.name = name

    return Z