import os
import sys
import numpy as np
import pandas as pd
from hsr.config import *
from matterhorn.data_loader.price_loader import PriceLoader
from matterhorn.universe.universe_loader import UniverseLoader
from matterhorn.util.tradingdays import TradingDays
import matterhorn.util.mputil as mputil
import statsmodels.api as sm


def _path_join(root, subfolder):
    res = os.path.join(root, subfolder)
    if not os.path.exists(res):
        os.makedirs(res)
    return res
    
_path_join(DEFAULT_PATH, "loading")

import numpy as np
import pandas as pd


def rolling_trend_over_mean_quarterly_np(y: np.ndarray, quarters: int = 20) -> np.ndarray:
    """
    Rolling descriptor for quarterly arrays (equal spacing):
        X_t = a + b * (t/4) + eps,  t = 0..T-1 (quarters)
        descriptor = b / mean(X)    where b is *per-year* slope.
    Parameters
    ----------
    y        : np.ndarray of shape (N,)   # quarterly levels (TA for AGRO, EPS for EGRO)
    quarters : int                        # window length (default 20 ≈ 5y)
    Returns
    -------
    np.ndarray of shape (N,) with np.nan for the first (quarters-1) entries.
    """
    y = np.asarray(y, dtype=float)
    N, T = y.size, quarters
    out = np.full(N, np.nan)

    if N < T or T < 3:
        return out

    # time regressor in YEARS with equal spacing: 0, 1/4, ..., (T-1)/4
    q = np.arange(T, dtype=float)
    t = q / 4.0

    # Precompute constants for OLS slope with fixed-length windows
    sum_t  = t.sum()
    sum_tt = (t * t).sum()
    denom  = T * sum_tt - sum_t * sum_t      # > 0 for T>=2

    # Rolling sums via sliding windows (vectorized)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        Y = sliding_window_view(y, T)                     # shape (N-T+1, T)
        sum_y  = Y.sum(axis=1)                            # (N-T+1,)
        sum_ty = (Y * t).sum(axis=1)                      # (N-T+1,)

        # per-year slope for each window
        b = (T * sum_ty - sum_t * sum_y) / denom          # (N-T+1,)
        m = sum_y / T
        val = np.where(m == 0.0, np.nan, b / m)

        out[T-1:] = val
        return out
    except Exception:
        # Fallback: simple loop (still quick for moderate N)
        sum_y = y[:T].sum()
        sum_ty = (y[:T] * t).sum()

        i_out = T - 1
        b = (T * sum_ty - sum_t * sum_y) / denom
        m = sum_y / T
        out[i_out] = np.nan if m == 0.0 else b / m

        for i in range(T, N):
            # slide window: drop y[i-T], add y[i]
            old, new = y[i - T], y[i]
            sum_y += new - old
            # sum_ty update: shift all y weights by +1 quarter:
            # new sum_ty = sum_{k=0}^{T-1} y[i-T+1+k] * (k/4)
            #            = (sum_ty - old*0/4) - (1/4)*sum_{k=0}^{T-2} y[i-T+1+k] + new*((T-1)/4)
            # Easiest (and still O(T)) is recompute; but keep it O(1) by storing full window if needed.
            # For simplicity here, recompute:
            win = y[i - T + 1 : i + 1]
            sum_ty = (win * t).sum()

            i_out += 1
            b = (T * sum_ty - sum_t * sum_y) / denom
            m = sum_y / T
            out[i_out] = np.nan if m == 0.0 else b / m

        return out


def build_gics(universe_data):
    # Industry dummies
    # # One-hot GICS (or your internal industry). If you don’t have GICS, you can approximate from 10-K NAICS/SIC—still “fundamentals-derived”.
    s = pd.Series(universe_data["Sector"].values, 
                  index=universe_data[identifier].values,
                  name="sector")
    sector_one_hot = pd.get_dummies(s, dtype="uint8", sparse=False)
    sector_one_hot.index.name = identifier
    sector_one_hot.columns.name = "Sector"
    out_fn = os.path.join(DEFAULT_PATH, "sector_one_hot.parquet")
    sector_one_hot.to_parquet(out_fn)
    print(f"Saved sector one hot to {out_fn}")


def _q_to_d(vals, qeds, all_dates):
    df = pd.Series(vals, index=qeds).dropna()
    df.index = pd.to_datetime(df.index)
    df = df[df.index >= start_date]
    df.index = TradingDays.shift_to_first_available_date_static(
        df.index.values, all_dates
    )
    df = TradingDays.with_dates(df, all_dates)
    df = df[~df.index.duplicated(keep='last')]
    return df.ffill()
    

def build_size(ticker, mkt_data, funda_data):
    print(f"Building Size and non-linear size for {ticker}")
    # Size: ln(Market Cap)
    close_df = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "close"]]
    close_df = close_df.set_index(date_col)["close"]

    nshares_df = _q_to_d(funda_data["IS_SH_FOR_DILUTED_EPS"].values,
                         funda_data["quarter_end_date"].values,
                         close_df.index)

    mcap_df = close_df * nshares_df
    size_df = np.log(mcap_df)
    size_df.name = "size"

    nonlinear_size_df = size_df ** 3
    nonlinear_size_df.name = "nonlinear_size"
    return pd.concat([size_df, nonlinear_size_df], axis=1)
    

def build_value(ticker, mkt_data, funda_data):
    print(f"Building Value for {ticker}")
    # Value (B/P): Book Equity/Price (lag BE by at least 3–6 months to avoid look-ahead).
    close_df = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "close"]]
    close_df = close_df.set_index(date_col)["close"]

    equities = funda_data["TOT_COMMON_EQY"] / funda_data["IS_SH_FOR_DILUTED_EPS"]
    equity_df = _q_to_d(equities.values,
                        funda_data["quarter_end_date"].values,
                        close_df.index)
    value_df = equity_df / close_df
    value_df.name = "value_bp"
    return value_df
    

def build_earnings_yield(ticker, mkt_data, funda_data):
    print(f"Building Earnings Yield for {ticker}")
    # Earnings Yield: Trailing 12mNI/Price.
    close_df = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "close"]]
    close_df = close_df.set_index(date_col)["close"]

    nis = funda_data["IS_NET_INC_AVAIL_COM_SHRHLDRS"] / funda_data["IS_SH_FOR_DILUTED_EPS"]
    ni_df = _q_to_d(nis.rolling(4).sum().values,
                    funda_data["quarter_end_date"].values,
                    close_df.index)

    ey_df = ni_df / close_df
    ey_df.name = "earnings_yield_trailing_12m"
    return ey_df


def build_growth(ticker, mkt_data, funda_data):
    print(f"Building Growth for {ticker}")
    # growth
    close_df = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "close"]]
    close_df = close_df.set_index(date_col)["close"]

    # 1. Payout ratio over five years
    payout_ratio = funda_data["IS_REGULAR_CASH_DIVIDEND_PER_SH"] / funda_data["IS_EPS"]
    payo_df = _q_to_d(payout_ratio.rolling(4*5).mean().values,
                    funda_data["quarter_end_date"].values,
                    close_df.index)
    payo_df.name = "growth_payo"

    # 2. Variability in capital structure
    n_diff = np.abs(funda_data["IS_SH_FOR_DILUTED_EPS"].diff())
    ld_diff = np.abs(funda_data["BS_LT_BORROW"].diff())
    pe_diff = np.abs(funda_data["PREFERRED_EQUITY_&_MINORITY_INT"].diff()).fillna(0.)
    n_diff_df = _q_to_d(n_diff.values, funda_data["quarter_end_date"].values, close_df.index)
    num = n_diff_df + _q_to_d(ld_diff.values+pe_diff.values,
                    funda_data["quarter_end_date"].values,
                    close_df.index)
    den = funda_data["TOT_COMMON_EQY"] + funda_data["BS_LT_BORROW"] + funda_data["PREFERRED_EQUITY_&_MINORITY_INT"]
    den = _q_to_d(den.values, funda_data["quarter_end_date"].values, close_df.index)
    vcap_df = num.rolling(252*5).mean() / den
    vcap_df.name = "growth_vcap"

    # 3. Growth rate in total assets
    agro = rolling_trend_over_mean_quarterly_np(funda_data["BS_TOT_ASSET"].values, 20)
    agro_df = _q_to_d(agro, funda_data["quarter_end_date"].values, close_df.index)
    agro_df.name = "growth_agro"

    # 4. Earnings growth rate over the last five years
    egro = rolling_trend_over_mean_quarterly_np(funda_data["IS_EPS"].values, 20)
    egro_df = _q_to_d(egro, funda_data["quarter_end_date"].values, close_df.index)
    egro_df.name = "growth_egro"

    # 5. Recent earnings change
    eps =  funda_data["IS_EPS"].rolling(4).sum()
    dele = (eps - eps.shift(4)) / (eps + eps.shift(4)) * 2
    dele_df = _q_to_d(dele.values, funda_data["quarter_end_date"].values, close_df.index)
    dele_df[dele_df < 0] = np.nan
    dele_df.name = "growth_dele"

    growth_df = pd.concat([payo_df, 
                           vcap_df,
                           agro_df,
                           egro_df, 
                           dele_df], axis=1)
    return growth_df


def build_leverage(ticker, mkt_data, funda_data):
    print(f"Building Leverage for {ticker}")
    close_df = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "close"]]
    close_df = close_df.set_index(date_col)["close"]

    # 1. Market leverage
    me = _q_to_d(funda_data["IS_SH_FOR_DILUTED_EPS"].values,
                 funda_data["quarter_end_date"].values,
                 close_df.index) * close_df
    num = funda_data["BS_LT_BORROW"].fillna(0.) + funda_data["PREFERRED_EQUITY_&_MINORITY_INT"].fillna(0.)
    num = _q_to_d(num.values,
                  funda_data["quarter_end_date"].values,
                  close_df.index) + me
    mlev_df = num / me
    mlev_df.name = "leverage_mlev"

    # 2. Book leverage
    num = (funda_da["TOT_COMMON_EQY"].fillna(0.) + \
           funda_data["BS_LT_BORROW"].fillna(0.) + \
           funda_data["PREFERRED_EQUITY_&_MINORITY_INT"].fillna(0.))
    den = funda_data["TOT_COMMON_EQY"].fillna(0.)
    blev_df = _q_to_d((num/den).values,
                    funda_data["quarter_end_date"].values,
                    close_df.index
                    )

    # 3. Debt to total assets
    debt = funda_data["BS_LT_BORROW"].fillna(0.) + funda_data["BS_ST_DEBT"].fillna(0.)
    asset = funda_data["BS_TOT_ASSET"]
    dtoa = debt / asset
    dtoa_df = _q_to_d(da.values,
                funda_data["quarter_end_date"].values,
                close_df.index)
    dtoa_df.name = "leverage_dtoa"

    leverage_df = pd.concat([mlev_df, blev_df, dtoa_df], axis=1)        
    return leverage_df


def build_momentum(ticker, mkt_data):
    # 1. Relative strength
    # 2. historical alpha
    print(f"Building Momentum for {ticker}")
    # Momentum (12–1): cumulative return over past 12 months excluding last month.
    adj_close = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "adj_close"]].set_index(date_col)
    simple_ret = adj_close["adj_close"] / adj_close["adj_close"].shift(1) - 1.
    mom_df = simple_ret.rolling(window=252).mean().shift(21)
    mom_df.name = "momentum_rs"
    return mom_df
    

def build_volatility(ticker, mkt_data):
    print(f"Building Volatility for {ticker}")
    # Beta times sigma
    # Daily standard deviation 
    adj_close = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "adj_close"]].set_index(date_col)
    simple_ret = adj_close["adj_close"] / adj_close["adj_close"].shift(1) - 1.
    std_df = simple_ret.rolling(window=60).std()
    std_df.name = "volatility_dsd"

    # High-low price
    # Log of stock price
    # Cumulative range
    # Volume beta 
    # Serial dependence 
    # Option-implied standard deviation
    volatility_df = std_df
    return volatility_df

def build_trading_activity(ticker, mkt_data, funda_data):
    print(f"Building Trading Activity for {ticker}")
    volume = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "volume", "close"]].set_index(date_col)
    volume = volume["volume"] / volume["close"]
    nshares = funda_data["IS_SH_FOR_DILUTED_EPS"]
    # Share turnover rate
    # 1. annual
    annual = _q_to_d(nshares.rolling(4).mean().values,
                    funda_data["quarter_end_date"].values,
                    volume.index)
    annual = volume.rolling(252).sum() / annual
    annual.name = "trading_activity_annual"

    # 2. quarter
    quarter = _q_to_d(nshares.values,
                    funda_data["quarter_end_date"].values,
                    volume.index)
    quarter = volume.rolling(63).sum() / quarter
    quarter.name = "trading_activity_quarter"

    # 3. five years
    five_year = _q_to_d(nshares.rolling(4*5).mean().values,
                    funda_data["quarter_end_date"].values,
                    volume.index)
    five_year = volume.rolling(252*5).sum() / five_year
    five_year.name = "trading_activity_5y"

    trading_activity_df = pd.concat([annual, quarter, five_year], axis=1)
    return trading_activity_df


def build_earnings_variability(ticker, mkt_data, funda_data):
    print(f"Building Earnings Variability for {ticker}")
    all_dates = pd.to_datetime(sorted(mkt_data[date_col].unique()))
    # 1. Variability in earnings
    earnings = funda_data["IS_NET_INC_AVAIL_COM_SHRHLDRS"]
    earnings = earnings.rolling(4).sum()
    den = earnings.rolling(4*5).mean()
    num = earnings.rolling(4*5).std()
    vern_df = _q_to_d((num / den).values,
                    funda_data["quarter_end_date"].values,
                    all_dates)
    vern_df.name = "earnings_variability_vern"
    earnings_variability_df = vern_df
    return earnings_variability_df
    

def build_dividend_yield(ticker, mkt_data, funda_data):
    print(f"Building Dividend Yield for {ticker}")
    close_df = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "close"]]
    close_df = close_df.set_index(date_col)["close"]

    dividend = funda_data["IS_REGULAR_CASH_DIVIDEND_PER_SH"]
    pr_df = _q_to_d(dividend.rolling(4).sum().values,
                    funda_data["quarter_end_date"].values,
                    close_df.index)

    dividend_yield_df = pr_df / close_df
    dividend_yield_df.name = "dividend_yield_annual"
    return dividend_yield_df


def calc_descriptor(ticker, mkt_data):
    funda_fn = os.path.join(FUNDA_PATH, f"{ticker.lower()}.csv")
    if not os.path.exists(funda_fn):
        print(f"Failed to load fundamentals for {ticker}")
        return

    funda_data = pd.read_csv(funda_fn).shift(1).ffill()
    dfs = []
    # dfs.append(build_volatility(ticker, mkt_data))
    # dfs.append(build_momentum(ticker, mkt_data))
    # dfs.append(build_size(ticker, mkt_data, funda_data))
    # dfs.append(build_trading_activity(ticker, mkt_data, funda_data))
    # dfs.append(build_growth(ticker, mkt_data, funda_data))
    # dfs.append(build_earnings_yield(ticker, mkt_data, funda_data))
    # dfs.append(build_value(ticker, mkt_data, funda_data))
    # dfs.append(build_earnings_variability(ticker, mkt_data, funda_data))
    dfs.append(build_leverage(ticker, mkt_data, funda_data))
    # dfs.append(build_dividend_yield(ticker, mkt_data, funda_data))

    df = pd.concat(dfs, axis=1)
    out_fn = os.path.join(DEFAULT_PATH, "descriptor", f"{ticker}.parquet")
    df.to_parquet(out_fn)
    print(f"Saved descriptors to {out_fn}")
    

def main():

    u = UniverseLoader(data_type=universe)
    build_gics(u.universe_df)

    loader = PriceLoader(region,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        universe,
                        identifier
                        )
    mkt_data = loader.load("price")

    tickers = mkt_data["Ticker"].unique()
    # tickers = ["CMDB"]
    for ticker in tickers:
        calc_descriptor(ticker, mkt_data)


if __name__ == "__main__":
    sys.exit(main())