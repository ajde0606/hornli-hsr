
import os
import sys
import numpy as np
import pandas as pd
from hsr.config import *
from hsr.loading import winsorize_and_standardize_descriptor, styles
from hsr.regression import cross_sectional_regression_one_day, concat_loadings
from hsr.analysis import compute_risk_from_panels_rolling, factor_variance_explained_per_asset


def _empty(S):
    for col in S.columns:
        if S[col].sum() == 0:
            return True
    return False


def run_regression():
    simple_ret_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "intermediate/simple_return.parquet"))
    industry_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "intermediate/industry_one_hot.parquet"))
    loading_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "intermediate/loadings.parquet")).reset_index()
    mkt_cap = pd.read_parquet(os.path.join(DEFAULT_PATH, "intermediate/cap_weights.parquet"))

    all_dates = np.intersect1d(simple_ret_df.index, loading_df[date_col].unique())
    all_tickers = np.intersect1d(simple_ret_df.columns, loading_df[identifier].unique())

    factor_returns = []
    specific_returns = []
    specific_variances = []
    for date in all_dates:
        r = simple_ret_df.loc[date, all_tickers].dropna()
        tickers_ = r.index
        D_ind = industry_df.loc[tickers_]
        X_style = loading_df[loading_df[date_col] == date].drop(columns=[date_col])
        X_style = X_style.set_index(identifier).loc[tickers_].fillna(0.)
        C_cty = pd.DataFrame(np.ones(len(tickers_)), index=tickers_, columns=["country"])
        mcap = mkt_cap.loc[date, tickers_].fillna(0.)

        if _empty(X_style): continue
        print(f"Processing {date}")

        f_ret, e = cross_sectional_regression_one_day(
            r, X_style, D_ind, C_cty, mcap,
            hsigma=prev_sv.pow(0.5) if 'prev_sv' in locals() else None,
        )

        f_ret.name = date
        factor_returns.append(f_ret)

        e.name = date
        specific_returns.append(e)

    factor_return = pd.concat(factor_returns, axis=1).T
    out_fn = os.path.join(DEFAULT_PATH, "output/factor_return.parquet")
    factor_return.to_parquet(out_fn)
    print(f"Factor return saved to {out_fn}")

    specific_return = pd.concat(specific_returns, axis=1).T
    out_fn = os.path.join(DEFAULT_PATH, "output/specific_return.parquet")
    specific_return.to_parquet(out_fn)
    print(f"Specific return saved to {out_fn}")


def compute_risk():
    factor_return_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "output/factor_return.parquet"))
    specific_return_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "output/specific_return.parquet"))
    regime_proxy = pd.read_csv(os.path.join(DEFAULT_PATH, "intermediate/sp.csv"),
                               header=None)
    regime_proxy = pd.Series(regime_proxy[1].values,
                             index=pd.to_datetime(regime_proxy[0].values),
                             name="regime")

    factor_cov, regime_multiplier, specific_var = compute_risk_from_panels_rolling(
        factor_return_df,
        specific_return_df.T,
        regime_proxy=regime_proxy,
        cov_format="long",
    )
    
    out_fn = os.path.join(DEFAULT_PATH, "output/factor_cov.parquet")
    factor_cov.to_parquet(out_fn)
    print(f"Factor covariance saved to {out_fn}")

    out_fn = os.path.join(DEFAULT_PATH, "output/specific_var.parquet")
    specific_var.T.to_parquet(out_fn)
    print(f"Specific var saved to {out_fn}")


def construct_loadings():
    ind = pd.read_parquet(os.path.join(DEFAULT_PATH, "intermediate/industry_one_hot.parquet"))
    tickers = ind.index

    descriptor_fns = glob.glob(os.path.join(DEFAULT_PATH, "intermediate/descriptor", "*.parquet"))
    descriptor_to_dfs = defaultdict(list)
    for fn in sorted(descriptor_fns):
        ticker = fn.split("/")[-1].split(".")[0]
        if ticker not in tickers: continue

        print("loading descriptors of ", ticker)
        df = pd.read_parquet(fn)
        for col in df.columns:
            df_ = df[col]
            df_.name = ticker
            descriptor_to_dfs[col].append(df_)

    zscore_dfs = []
    for descriptor, dfs in descriptor_to_dfs.items():
        # if "trading_activity" not in descriptor: continue
        
        if descriptor == "market_cap":
            out_fn = os.path.join(DEFAULT_PATH, "intermediate/cap_weights.parquet")
            pd.concat(dfs, axis=1).to_parquet(out_fn)
        else:
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

    out_fn = os.path.join(DEFAULT_PATH, "intermediate/loadings.parquet")
    loading_df.to_parquet(out_fn)
    print(f"Saved loadings to {out_fn}")


def research():
    loading_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "intermediate/loadings.parquet")).reset_index()
    cov_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "output/factor_cov.parquet"))
    spec_var_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "output/specific_var.parquet"))
    industry_df = pd.read_parquet(os.path.join(DEFAULT_PATH, "intermediate/industry_one_hot.parquet"))
    country_df = pd.DataFrame(1, index=industry_df.index, columns=["country"])

    r2s = []
    for date in spec_var_df.index:
        print(f"calculating {date}")

        X_s = loading_df[loading_df[date_col] == date].drop(columns=[date_col])
        X_s = X_s.set_index(identifier)

        Sigma_f_t = cov_df[cov_df["date"] == date]
        Sigma_f_t = Sigma_f_t.pivot(index="row_factor",
                                    columns="col_factor",
                                    values="value")
        spec_var_t = spec_var_df.loc[date]


        idx = spec_var_t.index
        X_t, all_cols = concat_loadings(X_s, industry_df, country_df, idx)
        X_t = pd.DataFrame(X_t, index=idx, columns=all_cols)

        r2 = factor_variance_explained_per_asset(X_t, Sigma_f_t, spec_var_t)
        r2.name = date
        r2s.append(r2)

    r2_df = pd.concat(r2s, axis=1).sort_index().T
    out_fn = os.path.join(DEFAULT_PATH, "output/r2.parquet")
    r2_df.to_parquet(out_fn)
    print(f"Saved r2 to {out_fn}")


def main():
    # construct_loadings()
    # run_regression()
    compute_risk()
    research()

    return 0

    
if __name__ == "__main__":
    sys.exit(main())