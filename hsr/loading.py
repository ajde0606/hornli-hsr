import os
import sys
import glob
import pandas as pd
from hsr.config import *
from collections import defaultdict


styles = [
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
        zscore_df = df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1), axis=0)
        zscore_df.clip(-3, 3, inplace=True)
        zscore_df.columns.name = identifier
        zscore_df.index.name = date_col
        zscore_df = zscore_df.unstack()
        zscore_df.name = descriptor
        zscore_dfs.append(zscore_df)
    zscore_df = pd.concat(zscore_dfs, axis=1)

    loading_dfs = []
    for style in styles:
        cols = [s for s in zscore_df.columns if s.startswith(style)]
        df = zscore_df[cols].mean(axis=1)
        df.name = style
        loading_dfs.append(df)
    loading_df = pd.concat(loading_dfs, axis=1)

    out_fn = os.path.join(DEFAULT_PATH, "loadings.parquet")
    loading_df.to_parquet(out_fn)
    print(f"Saved loadings to {out_fn}")


if __name__ == "__main__":
    sys.exit(main())