
import os
import sys
import numpy as np
import pandas as pd
from hsr.config import *
from hsr.regression import cross_sectional_risk_model_with_country


def empty(S):
    for col in S.columns:
        if S[col].sum() == 0:
            return True
    return False


def main():
    simple_ret_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "simple_return.parquet"))
    sector_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "sector_one_hot.parquet"))
    # XXX
    sectors = list(set(sector_df.columns) - set(["Financials", "Health Care", "Real Estate"]))
    sector_df = sector_df[sectors]
    loading_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "loadings.parquet")).reset_index()

    all_dates = np.intersect1d(simple_ret_df.index, loading_df[date_col].unique())
    all_tickers = np.intersect1d(simple_ret_df.columns, loading_df[identifier].unique())

    factor_returns = []
    specific_returns = []
    specific_variances = []
    for date in all_dates:
        # DEBUG!!
        # if pd.to_datetime(date) < pd.to_datetime("2025-01-01"): continue
        # r:        pd.Series of size N (monthly excess returns)
        # X_style:  pd.DataFrame (N x S) with columns like ['Value','Size','Momentum',...]
        # D_ind:    pd.DataFrame (N x K) one-hot industries (sum across K ~ 1 for each row)
        # prev_sv:  pd.Series of size N from yesterday (optional)
        r = simple_ret_df.loc[date, all_tickers].dropna()
        tickers_ = r.index
        D_ind = sector_df.loc[tickers_]
        X_style = loading_df[loading_df[date_col] == date].drop(columns=[date_col])
        X_style = X_style.set_index(identifier).loc[tickers_].fillna(0.)
        C_cty = pd.DataFrame(np.ones(len(tickers_)), index=tickers_, columns=["country"])

        if empty(X_style): continue
        print(f"Processing {date}")

        f_ret, u, sv = cross_sectional_risk_model_with_country(
            r, X_style, D_ind, C_cty,
            hsigma=prev_sv.pow(0.5) if 'prev_sv' in locals() else None,
            prev_spec_var=prev_sv if 'prev_sv' in locals() else None
        )

        # Outputs:
        # fr  -> pd.Series of factor returns keyed by factor names (styles then industries)
        # e   -> pd.Series of specific returns for each asset
        # sv  -> pd.Series of updated specific variances for each asset

        prev_sv = sv
    
        f_ret.name = date
        factor_returns.append(f_ret)

        u.name = date
        specific_returns.append(u)

        sv.name = date
        specific_variances.append(sv)

    factor_return = pd.concat(factor_returns, axis=1).T
    out_fn = os.path.join(DEFAULT_PATH, "factor_return.parquet")
    factor_return.to_parquet(out_fn)
    print(f"Factor return saved to {out_fn}")

    specific_return = pd.concat(specific_returns, axis=1).T
    out_fn = os.path.join(DEFAULT_PATH, "specific_return.parquet")
    specific_return.to_parquet(out_fn)
    print(f"Specific return saved to {out_fn}")

    specific_variance = pd.concat(specific_variances, axis=1).T
    out_fn = os.path.join(DEFAULT_PATH, "specific_variance.parquet")
    specific_variance.to_parquet(out_fn)
    print(f"Specific variance saved to {out_fn}")
    

if __name__ == "__main__":
    sys.exit(main())
        