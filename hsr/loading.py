import os
import sys
import glob
import numpy as np
import pandas as pd
from hsr.config import *
from collections import defaultdict

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

def winsorize_and_standardize_descriptor(
    df_desc: pd.DataFrame,                     # index=date, columns=tickers
    name: str,
    trim_sigma: float = 3.0
) -> pd.DataFrame:
    print("winsorizing and standardizing", name)
    X = df_desc.astype(float).copy()
    lower = X.quantile(0.001, axis=1)
    upper = X.quantile(0.999, axis=1)

    # clip row-wise by aligning on the index (axis=0)
    X = X.clip(lower=lower, upper=upper, axis=0)

    # 1) row-wise mean/std (equal-weighted)
    mu = X.mean(axis=1)
    sigma = X.std(axis=1, ddof=0)

    # 2) winsorize per date using row-wise clip (axis=0)
    lower_s = mu - trim_sigma * sigma   # Series indexed by date
    upper_s = mu + trim_sigma * sigma
    X = X.clip(lower=lower_s, upper=upper_s, axis=0)

    # 3) mean 0 per date
    w = pd.DataFrame(1.0, index=X.index, columns=X.columns)
    w = w.where(X.notna())
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)

    cap_mu = (X * w).sum(axis=1)
    Xc = X.sub(cap_mu, axis=0)

    # 4) equal-weighted stdev = 1 per date
    ew_std = Xc.std(axis=1, ddof=0).replace(0, np.nan)
    Z = Xc.div(ew_std, axis=0)

    Z.columns.name = identifier
    Z.index.name = date_col
    Z = Z.unstack()
    Z.name = name
    return Z