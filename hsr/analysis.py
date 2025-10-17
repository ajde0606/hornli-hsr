import numpy as np
import pandas as pd


def factor_variance_explained_per_asset(
    X_t: pd.DataFrame,          # N x K exposures (rows assets, cols factors) for day t
    Sigma_f_t: pd.DataFrame,    # K x K factor cov for day t
    spec_var_t: pd.Series       # N-vector of specific variances for day t (aligned to X_t.index)
) -> pd.Series:
    """
    Returns R2 per asset: fraction of total variance explained by factors.
    """
    X = X_t.astype(float).values               # (N,K)
    Sf = Sigma_f_t.astype(float).values        # (K,K)
    sv = spec_var_t.reindex(X_t.index).values  # (N,)

    # factor variance per asset: diag(X Sf X^T)
    XSf = X @ Sf                                # (N,K)
    fac_var = np.einsum('nk,nk->n', X, XSf)     # row-wise dot -> (N,)
    tot_var = fac_var + sv
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = np.where(tot_var > 0, fac_var / tot_var, 0.0)
    return pd.Series(r2, index=X_t.index, name="R2_factor_explained")


def factor_variance_explained_portfolio(
    w_t: pd.Series,             # N weights for day t
    X_t: pd.DataFrame,          # N x K exposures (same assets/order as w_t)
    Sigma_f_t: pd.DataFrame,    # K x K
    spec_var_t: pd.Series       # N specific variances
) -> tuple[float, float, float]:
    """
    Returns (R2_portfolio, var_factor, var_specific).
    """
    w = w_t.astype(float).reindex(X_t.index).fillna(0.0).values  # (N,)
    X = X_t.astype(float).values                                  # (N,K)
    Sf = Sigma_f_t.astype(float).values                           # (K,K)
    sv = spec_var_t.reindex(X_t.index).astype(float).values       # (N,)

    # portfolio factor exposures and factor variance
    beta_p = X.T @ w                          # (K,)
    var_factor = float(beta_p.T @ Sf @ beta_p)

    # portfolio specific variance
    var_specific = float((w**2 @ sv))

    tot = var_factor + var_specific
    r2_port = (var_factor / tot) if tot > 0 else 0.0
    return r2_port, var_factor, var_specific


def compute_risk_from_panels_rolling(
    factor_returns_df: pd.DataFrame,       # T x K, index=date asc, columns=factors
    specific_returns_df: pd.DataFrame,     # N x T, index=asset, columns=date asc
    *,
    # factor cov params
    hl_vol_days: float = 42.0,
    hl_corr_days: float = 200.0,
    hl_regime_days: float = 21.0,
    step_days: float = 1.0,
    ridge: float = 1e-10,
    shrink_to_diag: float = 0.0,
    regime_proxy: pd.Series | None = None,  # Series indexed by date; defaults to FR row mean
    # specific risk params
    spec_hl_vol_days: float = 42.0,
    spec_bayes_shrink: float = 0.05,
    specvar_clip_q: float | None = 0.995,
    # output format for Sigma
    cov_format: str = "dict"  # "dict" (date->DataFrame), "stacked" (MultiIndex columns), or "long"
):
    """
    Rolling daily risk:
      • Updates EWMA factor vols/corr and builds Σ_f(t) each day (with regime multiplier m(t))
      • Updates EWMA specific variances per asset from residuals e_i(t)^2 (with shrink + regime)

    Returns:
      Sigma_by_day, regime_mult_by_day, spec_var_by_day

      Sigma_by_day:
        - if cov_format=="dict": dict[date] -> KxK DataFrame
        - if cov_format=="stacked": DataFrame with columns MultiIndex (row_factor, col_factor), index=date
        - if cov_format=="long": DataFrame with columns [date, row_factor, col_factor, value]
      regime_mult_by_day: pd.Series indexed by date
      spec_var_by_day: DataFrame (N x T) indexed by asset, columns=date
    """
    # ---- prep & sort ----
    FR = factor_returns_df.sort_index().astype(float)
    dates = FR.index.to_list()
    K = FR.shape[1]
    fac_names = FR.columns

    SR = specific_returns_df.copy()
    SR = SR.loc[:, sorted(SR.columns)]  # ensure same order as FR
    if list(SR.columns) != dates:
        # intersect on dates to be safe
        common = [d for d in dates if d in SR.columns]
        FR = FR.loc[common]
        SR = SR.loc[:, common]
        dates = common

    # regime proxy series (per day)
    if regime_proxy is None:
        regime_series = FR.mean(axis=1)  # equal-weighted proxy
    else:
        regime_series = regime_proxy.reindex(dates).astype(float).fillna(0.0)

    # ---- lambdas ----
    lam_vol   = _lambda_from_half_life_days(hl_vol_days,   step_days=step_days)
    lam_corr  = _lambda_from_half_life_days(hl_corr_days,  step_days=step_days)
    lam_short = _lambda_from_half_life_days(hl_regime_days,step_days=step_days)
    lam_long  = _lambda_from_half_life_days(hl_vol_days,   step_days=step_days)
    lam_spec  = _lambda_from_half_life_days(spec_hl_vol_days, step_days=step_days)

    # ---- state ----
    var_k = np.zeros(K)               # EWMA factor variances
    cw = np.zeros((K, K))             # EWMA corr kernel on standardized factors
    v_short = 0.0; v_long = 0.0       # regime variances (proxy)
    # specific variance state (per asset)
    assets = SR.index
    spec_var_state = pd.Series(0.0, index=assets, dtype=float)

    # ---- outputs ----
    regime_mult_by_day = pd.Series(index=dates, dtype=float)
    Sigma_store_dict: dict[pd.Timestamp, pd.DataFrame] = {}
    Sigma_rows = []  # for "stacked"/"long"
    spec_var_by_day = pd.DataFrame(index=assets, columns=dates, dtype=float)

    # prebuild MultiIndex for "stacked"/"long"
    stacked_cols = pd.MultiIndex.from_product([fac_names, fac_names], names=["row_factor","col_factor"])

    # ---- roll forward day by day ----
    for d in dates:
        # ----- factor step -----
        x = FR.loc[d].values  # factor returns @ day d
        x = np.nan_to_num(x)

        # update EWMA factor variances
        var_k = lam_vol * var_k + (1.0 - lam_vol) * (x * x)
        vol_k = np.sqrt(np.clip(var_k, 1e-16, None))

        # update EWMA pairwise corr kernel on standardized returns
        z = np.divide(x, np.where(vol_k > 0, vol_k, 1.0), out=np.zeros_like(x), where=True)
        cw = lam_corr * cw + (1.0 - lam_corr) * np.outer(z, z)

        # turn kernel into correlation
        diag = np.clip(np.diag(cw), 1e-16, None)
        inv_s = 1.0 / np.sqrt(diag)
        corr = (inv_s[:, None] * cw) * inv_s[None, :]
        corr = np.clip(corr, -1.0, 1.0)

        # regime update
        rp = float(regime_series.loc[d])
        r2 = rp * rp
        v_short = lam_short * v_short + (1.0 - lam_short) * r2
        v_long  = lam_long  * v_long  + (1.0 - lam_long)  * r2
        regime_mult = float(np.sqrt(np.clip(v_short / max(v_long, 1e-16), 1e-4, 1e4)))
        regime_mult_by_day.loc[d] = regime_mult

        # covariance for the day
        D = np.diag(vol_k * regime_mult)
        Sigma = D @ corr @ D
        if shrink_to_diag > 0.0:
            Sigma = (1.0 - shrink_to_diag) * Sigma + shrink_to_diag * np.diag(np.diag(Sigma))
        Sigma = Sigma + ridge * np.eye(K)

        # store Sigma
        if cov_format == "dict":
            Sigma_store_dict[d] = pd.DataFrame(Sigma, index=fac_names, columns=fac_names)
        elif cov_format == "stacked":
            Sigma_rows.append(pd.Series(Sigma.ravel(), index=stacked_cols, name=d))
        elif cov_format == "long":
            # defer; we’ll build from "stacked" path
            Sigma_rows.append(pd.Series(Sigma.ravel(), index=stacked_cols, name=d))
        else:
            raise ValueError('cov_format must be one of {"dict","stacked","long"}')

        # ----- specific step -----
        e = SR[d].astype(float)                       # residuals for day d (per asset)
        e2 = e.pow(2)
        if specvar_clip_q is not None and 0.5 < specvar_clip_q < 1.0:
            q = float(np.nanquantile(e2.values, specvar_clip_q))
            e2 = e2.clip(upper=q)
        e2 = e2.fillna(e2.median()).clip(lower=1e-12)

        # EWMA update
        spec_var_state = lam_spec * spec_var_state + (1.0 - lam_spec) * e2

        # cross-sectional Bayesian shrink (per-day)
        pool = float(spec_var_state.median())
        sv = (1.0 - spec_bayes_shrink) * spec_var_state + spec_bayes_shrink * pool

        # link regime
        sv = sv * (regime_mult ** 2)

        # store
        spec_var_by_day[d] = sv.values

    # finalize Sigma outputs if stacked/long
    if cov_format == "stacked":
        Sigma_by_day = pd.DataFrame(Sigma_rows).sort_index()
    elif cov_format == "long":
        stacked = pd.DataFrame(Sigma_rows).sort_index()
        Sigma_by_day = (
            stacked.stack(["row_factor","col_factor"])
                   .rename("value")
                   .reset_index()
                   .rename(columns={"level_0":"date"})
        )
    else:
        Sigma_by_day = Sigma_store_dict

    spec_var_by_day = spec_var_by_day.astype(float)

    return Sigma_by_day, regime_mult_by_day.astype(float), spec_var_by_day
