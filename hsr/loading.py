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
    cap_weights: pd.DataFrame | pd.Series | None = None,
    trim_sigma: float = 3.0
) -> pd.DataFrame:
    print("winsorizing and standardizing", name)
    X = df_desc.astype(float).copy()

    # 1) row-wise mean/std (equal-weighted)
    mu = X.mean(axis=1)
    sigma = X.std(axis=1, ddof=0)

    # 2) winsorize per date using row-wise clip (axis=0)
    lower_s = mu - trim_sigma * sigma   # Series indexed by date
    upper_s = mu + trim_sigma * sigma
    X = X.clip(lower=lower_s, upper=upper_s, axis=0)

    # 3) cap-weighted mean 0 per date
    if cap_weights is None:
        w = pd.DataFrame(1.0, index=X.index, columns=X.columns)
    else:
        if isinstance(cap_weights, pd.Series) and cap_weights.index.equals(X.columns):
            # static weights by ticker -> broadcast to all dates
            w = pd.DataFrame([cap_weights], index=[X.index[0]]).reindex(X.index).ffill()
        else:
            w = pd.DataFrame(cap_weights, index=X.index, columns=X.columns, dtype=float)

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


def main():
    descriptor_fns = glob.glob(os.path.join(DEFAULT_PATH, "descriptor", "*.parquet"))
    descriptor_to_dfs = defaultdict(list)
    for fn in sorted(descriptor_fns):
        ticker = fn.split("/")[-1].split(".")[0]
        print("loading descriptors of ", ticker)
        df = pd.read_parquet(fn)
        for col in df.columns:
            df_ = df[col]
            df_.name = ticker
            descriptor_to_dfs[col].append(df_)
    
    zscore_dfs = []
    for descriptor, dfs in descriptor_to_dfs.items():
        df = pd.concat(dfs, axis=1)
        zscore_df = winsorize_and_standardize_descriptor(df, descriptor)
        zscore_dfs.append(zscore_df)
    zscore_df = pd.concat(zscore_dfs, axis=1)

    loading_dfs = []
    for style in styles:
        cols = [s for s in zscore_df.columns if s.startswith(style)]
        df = zscore_df[cols].mean(axis=1)
        df.name = style
        df = df.reset_index().pivot(index=date_col, columns=identifier, values=style)
        df = winsorize_and_standardize_descriptor(df, style)
        loading_dfs.append(df)
    loading_df = pd.concat(loading_dfs, axis=1)

    out_fn = os.path.join(DEFAULT_PATH, "loadings.parquet")
    loading_df.to_parquet(out_fn)
    print(f"Saved loadings to {out_fn}")


if __name__ == "__main__":
    sys.exit(main())