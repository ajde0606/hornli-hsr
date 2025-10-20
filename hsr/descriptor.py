import os
import sys
import numpy as np
import pandas as pd
from hsr.config import *


def _path_join(root, subfolder):
    res = os.path.join(root, subfolder)
    if not os.path.exists(res):
        os.makedirs(res)
    return res


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


def calc_return(adj_close, window):
    return np.log(adj_close / adj_close.shift(window)) / window


def build_gics(industry_df):
    # Industry dummies
    industry_df.dropna(inplace=True)
    industry_df = industry_df[industry_df["GICS_INDUSTRY_NAME"] != "#N/A Invalid Security"]
    tickers = [s.split()[0] for s in industry_df[identifier].values]

    s = pd.Series(industry_df["GICS_INDUSTRY_NAME"].values, 
                  index=tickers,
                  name="industry")
    sector_one_hot = pd.get_dummies(s, dtype="uint8", sparse=False)
    sector_one_hot.index.name = identifier
    sector_one_hot.columns.name = "industry"
    out_fn = os.path.join(DEFAULT_PATH, "intermediate/industry_one_hot.parquet")
    sector_one_hot.to_parquet(out_fn)
    print(f"Saved industry one hot to {out_fn}")


def _shift_to_first_available_date_static(dts, all_dates):
    res = dts.copy()
    for i, dt in enumerate(dts):
        if dt in all_dates: continue
        idx = np.where(all_dates>=dt)[0][0]
        res[i] = all_dates[idx]
    return res
    

def _with_dates(df, dates, how='right'):
    """df resampled at dates."""

    is_series = isinstance(df, pd.Series)
    if is_series:
        df = pd.DataFrame(df)

    res_df = pd.DataFrame(dates.values, columns=['date'])
    res_df = res_df.set_index('date')
    res_df = df.join(res_df, how=how)

    if is_series:
        res_df = res_df.iloc[:, 0]

    return res_df


def _q_to_d(vals, qeds, all_dates):
    df = pd.Series(vals, index=qeds).dropna()
    df.index = pd.to_datetime(df.index)
    df = df[df.index >= start_date]
    df.index = _shift_to_first_available_date_static(
        df.index.values, all_dates
    )
    df = _with_dates(df, all_dates)
    df = df[~df.index.duplicated(keep='last')]
    return df.ffill()


def build_market_cap(ticker, mkt_data, funda_data):
    print(f"Building Market Cap for {ticker}")
    close_df = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "close"]]
    close_df = close_df.set_index(date_col)["close"]

    nshares_df = _q_to_d(funda_data["Diluted Weighted Average Shares"].values,
                         funda_data.index,
                         close_df.index)

    mcap_df = close_df * nshares_df
    mcap_df.name = "market_cap"
    return mcap_df
    

def build_size(ticker, mkt_data, funda_data):
    print(f"Building Size and non-linear size for {ticker}")
    # Size: ln(Market Cap)
    close_df = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "close"]]
    close_df = close_df.set_index(date_col)["close"]

    nshares_df = _q_to_d(funda_data["Diluted Weighted Average Shares"].values,
                         funda_data.index,
                         close_df.index)

    mcap_df = close_df * nshares_df
    size_df = np.log(mcap_df)
    size_df.name = "size"

    nonlinear_size_df = size_df ** 3
    nonlinear_size_df.name = "nonlinear_size"
    return pd.concat([size_df, nonlinear_size_df], axis=1)
    

def build_value(ticker, mkt_data, funda_data):
    print(f"Building Value for {ticker}")
    # B/P
    close_df = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "close"]]
    close_df = close_df.set_index(date_col)["close"]

    equities = funda_data["Total Common Equity"] / funda_data["Diluted Weighted Average Shares"]
    equity_df = _q_to_d(equities.values,
                        funda_data.index,
                        close_df.index)
    bp_df = equity_df / close_df
    bp_df.name = "value_bp"

    # Sales/P
    sales = funda_data["Revenue"].rolling(4).sum() / funda_data["Diluted Weighted Average Shares"].rolling(4).mean()
    sales_df = _q_to_d(sales.values,
                       funda_data.index,
                       close_df.index)
    sp_df = sales_df / close_df
    sp_df.name = "value_sp"

    # CF/P
    cf = funda_data["Cash From Operations"].rolling(4).sum() / funda_data["Diluted Weighted Average Shares"].rolling(4).mean()
    cfp_df = _q_to_d(cf.values,
                    funda_data.index,
                    close_df.index) / close_df
    cfp_df.name = "value_cfp"

    value_df = pd.concat([bp_df, sp_df, cfp_df], axis=1)
    return value_df
    

def build_earnings_yield(ticker, mkt_data, funda_data):
    print(f"Building Earnings Yield for {ticker}")
    # Earnings Yield: Trailing 12mNI/Price.
    close_df = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "close"]]
    close_df = close_df.set_index(date_col)["close"]

    nis = funda_data["Net Income Available To Common Shareholders - IS"] / funda_data["Diluted Weighted Average Shares"]
    ni_df = _q_to_d(nis.rolling(4).sum().values,
                    funda_data.index,
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
    payout_ratio = funda_data["Regular Cash Dividend Per Share"] / funda_data["Basic Earnings per Share"]
    payo_df = _q_to_d(payout_ratio.rolling(4*5).mean().values,
                    funda_data.index,
                    close_df.index)
    payo_df.name = "growth_payo"

    # 2. Variability in capital structure
    n_diff = np.abs(funda_data["Diluted Weighted Average Shares"].diff())
    ld_diff = np.abs(funda_data["Long Term Debt"].diff())
    pe_diff = np.abs(funda_data["Preferred Equity and Minority Interest"].diff()).fillna(0.)
    n_diff_df = _q_to_d(n_diff.values, funda_data.index, close_df.index)
    num = n_diff_df + _q_to_d(ld_diff.values+pe_diff.values,
                    funda_data.index,
                    close_df.index)
    den = funda_data["Total Common Equity"] + funda_data["Long Term Debt"] + funda_data["Preferred Equity and Minority Interest"]
    den = _q_to_d(den.values, funda_data.index, close_df.index)
    vcap_df = num.rolling(252*5).mean() / den
    vcap_df.name = "growth_vcap"

    # 3. Growth rate in total assets
    sales = funda_data["Total Assets"] / funda_data["Diluted Weighted Average Shares"]
    agro = rolling_trend_over_mean_quarterly_np(sales.values, 20)
    agro_df = _q_to_d(agro, funda_data.index, close_df.index)
    agro_df.name = "growth_agro"

    # 4. Earnings growth rate over the last five years
    egro = rolling_trend_over_mean_quarterly_np(funda_data["Basic Earnings per Share"].values, 20)
    egro_df = _q_to_d(egro, funda_data.index, close_df.index)
    egro_df.name = "growth_egro"

    # 5. Recent earnings change
    eps =  funda_data["Basic Earnings per Share"].rolling(4).sum()
    dele = (eps - eps.shift(4)) / (eps + eps.shift(4)) * 2
    dele_df = _q_to_d(dele.values, funda_data.index, close_df.index)
    dele_df[dele_df < 0] = 0.
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
    me = _q_to_d(funda_data["Diluted Weighted Average Shares"].values,
                 funda_data.index,
                 close_df.index) * close_df
    num = funda_data["Long Term Debt"] + funda_data["Preferred Equity and Minority Interest"].fillna(0.)
    num = _q_to_d(num.values,
                  funda_data.index,
                  close_df.index) + me
    mlev_df = num / me
    mlev_df.name = "leverage_mlev"

    # 2. Book leverage
    num = (funda_data["Total Common Equity"] + \
           funda_data["Long Term Debt"] + \
           funda_data["Preferred Equity and Minority Interest"].fillna(0.))
    den = funda_data["Total Common Equity"]
    blev_df = _q_to_d((num/den).values,
                    funda_data.index,
                    close_df.index
                    )
    blev_df.name = "leverage_blev"

    # 3. Debt to total assets
    debt = funda_data["Long Term Debt"] + funda_data["Short Term Debt"].fillna(0.)
    asset = funda_data["Total Assets"]
    dtoa = debt / asset
    dtoa_df = _q_to_d(dtoa.values,
                funda_data.index,
                close_df.index)
    dtoa_df.name = "leverage_dtoa"

    leverage_df = pd.concat([mlev_df, blev_df, dtoa_df], axis=1)        
    return leverage_df


def build_momentum(ticker, mkt_data):
    # 1. Relative strength
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
    high = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "high"]].set_index(date_col)
    high = high["high"].rolling(window=21).max()
    low = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "low"]].set_index(date_col)
    low = low["low"].rolling(window=21).min()
    hilo_df = np.log(high/low)
    hilo_df.name = "volatility_hilo"

    # Log of stock price
    # Cumulative range
    # Volume beta 
    # Serial dependence 
    # Option-implied standard deviation
    volatility_df = pd.concat([std_df, hilo_df], axis=1)
    return volatility_df

def build_trading_activity(ticker, mkt_data, funda_data):
    print(f"Building Trading Activity for {ticker}")
    volume = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "volume", "close"]].set_index(date_col)
    volume = volume["volume"] / volume["close"]
    nshares = funda_data["Diluted Weighted Average Shares"]
    # Share turnover rate
    # 1. annual
    annual = _q_to_d(nshares.rolling(4).mean().values,
                    funda_data.index,
                    volume.index)
    annual = volume.rolling(252).sum() / annual
    annual.name = "trading_activity_annual"

    # 2. quarter
    quarter = _q_to_d(nshares.values,
                    funda_data.index,
                    volume.index)
    quarter = volume.rolling(63).sum() / quarter
    quarter.name = "trading_activity_quarter"

    # 3. five years
    five_year = _q_to_d(nshares.rolling(4*5).mean().values,
                    funda_data.index,
                    volume.index)
    five_year = volume.rolling(252*5).sum() / five_year
    five_year.name = "trading_activity_5y"

    trading_activity_df = pd.concat([annual, quarter, five_year], axis=1)
    return trading_activity_df


def build_earnings_variability(ticker, mkt_data, funda_data):
    print(f"Building Earnings Variability for {ticker}")
    all_dates = pd.to_datetime(sorted(mkt_data[date_col].unique()))
    # 1. Variability in earnings
    earnings = funda_data["Net Income Available To Common Shareholders - IS"]
    earnings = earnings.rolling(4).sum()
    den = earnings.rolling(4*5).mean()
    num = earnings.rolling(4*5).std()
    vern_df = _q_to_d((num / den).values,
                    funda_data.index,
                    all_dates)
    vern_df.name = "earnings_variability_vern"
    earnings_variability_df = vern_df

    # 2. Variability in sales
    sales = funda_data["Revenue"]
    sales = sales.rolling(4).sum()
    den = sales.rolling(4*5).mean()
    num = sales.rolling(4*5).std()
    vsales_df = _q_to_d((num / den).values,
                    funda_data.index,
                    all_dates)
    vsales_df.name = "earnings_variability_vsales"

    # 3. Variability in cash from operations
    cfs = funda_data["Cash From Operations"]
    cfs = cfs.rolling(4).sum()
    den = cfs.rolling(4*5).mean()
    num = cfs.rolling(4*5).std()
    vcfs_df = _q_to_d((num / den).values,
                    funda_data.index,
                    all_dates)
    vcfs_df.name = "earnings_variability_vcfs"

    # 4. Accruals using cash-flow statement
    ni = funda_data["Net Income Available To Common Shareholders - IS"].rolling(4).sum()
    cfo = funda_data["Cash From Operations"].rolling(4).sum()
    ta = funda_data["Total Assets"].rolling(4).mean()
    cfaccruals_df = _q_to_d(((ni - cfo) /ta).values,
                    funda_data.index,
                    all_dates)
    cfaccruals_df.name = "earnings_variability_cfaccruals"

    earnings_variability_df = pd.concat([earnings_variability_df, 
                                         vsales_df, 
                                         vcfs_df, 
                                         cfaccruals_df], axis=1)
    return earnings_variability_df
    

def build_dividend_yield(ticker, mkt_data, funda_data):
    print(f"Building Dividend Yield for {ticker}")
    close_df = mkt_data.loc[mkt_data[identifier] == ticker, [date_col, "close"]]
    close_df = close_df.set_index(date_col)["close"]

    dividend = funda_data["Regular Cash Dividend Per Share"]
    pr_df = _q_to_d(dividend.rolling(4).sum().values,
                    funda_data.index,
                    close_df.index)

    dividend_yield_df = pr_df / close_df
    dividend_yield_df.name = "dividend_yield_annual"
    return dividend_yield_df


def build_beta(ticker, mkt_data, sp_df):
    halflife = 63                          # ~3 months of trading days
    min_periods = 60                       # warm-up
    price_col = "adj_close"
    ticker_df = mkt_data.loc[mkt_data[identifier] == ticker].set_index(date_col)

    # 1) Align and create daily simple returns
    px = pd.DataFrame({
        "stk": ticker_df[price_col].astype(float),
        "mkt": sp_df.astype(float),
    }).dropna()
    r = px.pct_change().dropna()

    # 2) Excess returns (optional)
    r_ex_stk = r["stk"]
    r_ex_mkt = r["mkt"]

    # 3) EWM covariance/variance (EWM-weighted market model slope)
    #    Beta_t = Cov_t(stk, mkt) / Var_t(mkt)
    ewm_cov = r_ex_stk.ewm(halflife=halflife, min_periods=min_periods, adjust=False).cov(r_ex_mkt)
    ewm_var = r_ex_mkt.ewm(halflife=halflife, min_periods=min_periods, adjust=False).var()
    beta = (ewm_cov / ewm_var).dropna()

    return beta.rename("beta")


def build_management_quality(ticker, mkt_data, funda_data):
    print(f"Building Management Quality for {ticker}")
    all_dates = pd.to_datetime(sorted(mkt_data[date_col].unique()))
    # asset growth
    asset = funda_data["Total Assets"].rolling(4).mean()
    ag = asset / asset.shift(4) - 1.
    ag_df = _q_to_d(ag.values,
                    funda_data.index,
                    all_dates)
    ag_df.name = "management_quality_ag"

    # issuance growth
    shares = funda_data["Diluted Weighted Average Shares"].rolling(4).mean()
    sg = shares / shares.shift(4) - 1.
    sg_df = _q_to_d(sg.values,
                    funda_data.index,
                    all_dates)
    sg_df.name = "management_quality_sg"

    # capital expenditure growth
    capex = funda_data["Capital Expenditures - Absolute Value"].rolling(4).sum()
    cg = capex / capex.shift(4) - 1.
    cg_df = _q_to_d(cg.values,
                    funda_data.index,
                    all_dates)
    cg_df[cg_df == np.inf] = 0
    cg_df.name = "management_quality_cg"

    # capital expenditure
    ca = capex / asset
    ca_df = _q_to_d(ca.values,
                    funda_data.index,
                    all_dates)
    ca_df.name = "management_quality_ca"

    management_quality_df = pd.concat([ag_df, sg_df, cg_df, ca_df], axis=1)
    return management_quality_df


def calc_descriptor_for_one_stock(ticker, mkt_data, sp_df):
    funda_fn = os.path.join(FUNDA_PATH, f"{ticker.lower()}.csv")
    if not os.path.exists(funda_fn):
        print(f"Failed to load fundamentals for {ticker}")
        return

    out_fn = os.path.join(DEFAULT_PATH, "intermediate/descriptor", f"{ticker}.parquet")

    # # DEBUG
    # if os.path.exists(out_fn):
    #     return

    funda_data = pd.read_csv(funda_fn).set_index("Dates")
    last_day = pd.to_datetime(funda_data.index[-1]) + pd.Timedelta(days=91)
    if last_day < pd.Timestamp.now():
        funda_data.loc[last_day.strftime("%Y-%m-%d")] = np.nan
    funda_data = funda_data.shift(1).ffill()

    dfs = []
    dfs.append(build_beta(ticker, mkt_data, sp_df))
    dfs.append(build_volatility(ticker, mkt_data))
    dfs.append(build_momentum(ticker, mkt_data))
    dfs.append(build_size(ticker, mkt_data, funda_data))
    dfs.append(build_trading_activity(ticker, mkt_data, funda_data))
    dfs.append(build_growth(ticker, mkt_data, funda_data))
    dfs.append(build_earnings_yield(ticker, mkt_data, funda_data))
    dfs.append(build_value(ticker, mkt_data, funda_data))
    dfs.append(build_earnings_variability(ticker, mkt_data, funda_data))
    dfs.append(build_leverage(ticker, mkt_data, funda_data))
    dfs.append(build_dividend_yield(ticker, mkt_data, funda_data))
    dfs.append(build_management_quality(ticker, mkt_data, funda_data))
    dfs.append(build_market_cap(ticker, mkt_data, funda_data))

    df = pd.concat(dfs, axis=1)
    df.to_parquet(out_fn)
    print(f"Saved descriptors to {out_fn}")
    