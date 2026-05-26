import os
from pathlib import Path

import numpy as np
import pandas as pd


def _infer_periods_per_year(freq=None, R=None):
    if freq == "week":
        return 52
    if freq == "month":
        return 12
    if freq == "quarter":
        return 4
    if R in {5}:
        return 52
    if R in {20, 21, 22}:
        return 12
    if R in {60, 65, 66}:
        return 4
    return 252 / R if R else 252


def make_prediction_frame(meta, y_true, pred_prob, pred_label_threshold=0.5):
    """
    Combine test metadata, true labels, and model probabilities into one table.

    Parameters
    ----------
    meta : pandas.DataFrame
        Usually meta_test.csv loaded as a DataFrame.
    y_true : array-like
        True test labels, aligned with meta rows.
    pred_prob : array-like
        Predicted probability of label=1, aligned with meta rows.
    pred_label_threshold : float
        Probability threshold used to create pred_label.
    """
    pred_df = meta.copy()
    pred_df["y_true"] = np.asarray(y_true, dtype=np.int64)
    pred_df["pred_prob"] = np.asarray(pred_prob, dtype=float)
    pred_df["pred_label"] = (pred_df["pred_prob"] > pred_label_threshold).astype(int)
    pred_df["correct"] = pred_df["pred_label"] == pred_df["y_true"]

    for col in ["start_date", "date", "label_end_date"]:
        if col in pred_df.columns:
            pred_df[col] = pd.to_datetime(pred_df[col], errors="coerce")

    pred_df["ticker"] = pred_df["ticker"].astype(str)
    pred_df["ret"] = pd.to_numeric(pred_df["ret"], errors="coerce")
    pred_df = pred_df.dropna(subset=["date", "ticker", "pred_prob", "ret"])
    return pred_df.sort_values(["date", "ticker"]).reset_index(drop=True)


def assign_decile_groups(pred_df, n_groups=10):
    """
    Assign stocks into JS_cnn-style prediction-probability groups per rebalance date.

    Group 1 is the lowest predicted up-probability group.
    Group n_groups is the highest predicted up-probability group.

    JS_cnn cuts directly on raw probability percentiles, rather than ranking first.
    The first bucket includes its lower bound, while later buckets are open on the
    lower bound and closed on the upper bound.
    """
    out = []
    for date, group in pred_df.groupby("date", sort=True):
        group = group.dropna(subset=["pred_prob", "ret"]).copy()

        pred_prob = group["pred_prob"]
        low_decile = np.percentile(pred_prob, 100.0 / n_groups)
        high_decile = np.percentile(pred_prob, (n_groups - 1) * 100.0 / n_groups)
        if low_decile == high_decile:
            continue

        group["portfolio_group"] = np.nan
        for decile_idx in range(n_groups):
            low = np.percentile(pred_prob, decile_idx * 100.0 / n_groups)
            high = np.percentile(pred_prob, (decile_idx + 1) * 100.0 / n_groups)
            if decile_idx == 0:
                mask = (pred_prob >= low) & (pred_prob <= high)
            else:
                mask = (pred_prob > low) & (pred_prob <= high)
            group.loc[mask, "portfolio_group"] = decile_idx + 1

        group = group.dropna(subset=["portfolio_group"]).copy()
        group["portfolio_group"] = group["portfolio_group"].astype(int)
        out.append(group)

    if not out:
        return pd.DataFrame(columns=list(pred_df.columns) + ["portfolio_group"])

    return pd.concat(out, ignore_index=True)


def _find_jasper_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()
    for candidate in [start] + list(start.parents):
        if (candidate / "CACHE_DIR").exists() and (candidate / "Reimage_price_trends").exists():
            return candidate
    return None


def _default_marketcap_path():
    jasper_root = _find_jasper_root()
    if jasper_root is None:
        return None
    for name in ("us_week_ret.pq", "us_week_ret.parquet", "us_week_ret.csv"):
        path = jasper_root / "CACHE_DIR" / name
        if path.exists():
            return path
    return None


def add_marketcap_from_us_week_ret(grouped_df, marketcap_path=None):
    """
    Add JS_cnn MarketCap to Reimage portfolio assignments.

    The JS_cnn cache uses Date/StockID, while Reimage uses date/ticker.
    """
    if grouped_df.empty:
        return grouped_df.copy()

    if marketcap_path is None:
        marketcap_path = _default_marketcap_path()
    if marketcap_path is None:
        raise FileNotFoundError(
            "Could not find CACHE_DIR/us_week_ret.pq. Pass marketcap_path explicitly."
        )

    marketcap_path = Path(marketcap_path)
    if marketcap_path.suffix.lower() in {".pq", ".parquet"}:
        marketcap = pd.read_parquet(marketcap_path, columns=["Date", "StockID", "MarketCap"])
    else:
        marketcap = pd.read_csv(marketcap_path, usecols=["Date", "StockID", "MarketCap"])

    marketcap = marketcap.rename(
        columns={"Date": "date", "StockID": "ticker", "MarketCap": "MarketCap"}
    )
    marketcap["date"] = pd.to_datetime(marketcap["date"], errors="coerce")
    marketcap["ticker"] = marketcap["ticker"].astype(str)
    marketcap["MarketCap"] = pd.to_numeric(marketcap["MarketCap"], errors="coerce").abs()
    marketcap = marketcap.dropna(subset=["date", "ticker", "MarketCap"])
    marketcap = marketcap.drop_duplicates(subset=["date", "ticker"], keep="last")

    out = grouped_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str)
    out = out.drop(columns=["MarketCap"], errors="ignore")
    return out.merge(marketcap, on=["date", "ticker"], how="left")


def _assign_marketcap_transaction_bps(
    grouped_df,
    base_bps=1.0,
    mid_bps=5.0,
    small_bps=10.0,
    large_quantile=0.8,
    mid_quantile=0.4,
):
    """
    Assign one-way transaction-cost bps by date-level MarketCap ranks.

    Larger stocks get base_bps, middle stocks get mid_bps, and smaller stocks get
    small_bps. Missing MarketCap falls back to small_bps.
    """
    out = grouped_df.copy()
    if "MarketCap" not in out.columns:
        out["transaction_cost_bps"] = float(base_bps)
        return out

    out["MarketCap"] = pd.to_numeric(out["MarketCap"], errors="coerce").abs()
    cap_rank = out.groupby("date")["MarketCap"].rank(pct=True)
    out["transaction_cost_bps"] = np.select(
        [
            cap_rank >= large_quantile,
            cap_rank >= mid_quantile,
        ],
        [
            float(base_bps),
            float(mid_bps),
        ],
        default=float(small_bps),
    )
    out.loc[out["MarketCap"].isna(), "transaction_cost_bps"] = float(small_bps)
    return out


def _membership_rebalance_cost(sub, prev_assets, one_way_bps):
    """
    Compute membership-based rebalance cost for an equal-weight long portfolio.

    Stocks that remain in the group are not charged. Newly bought and sold names
    are charged by their equal-weight notional share.
    """
    curr_assets = set(sub["ticker"].astype(str))
    if len(curr_assets) == 0:
        return 0.0, 0.0, curr_assets

    if prev_assets is None:
        bought = curr_assets
        sold = set()
    else:
        bought = curr_assets - prev_assets
        sold = prev_assets - curr_assets

    curr_size = len(curr_assets)
    prev_size = len(prev_assets) if prev_assets else curr_size
    bought_weight = len(bought) / curr_size if curr_size else 0.0
    sold_weight = len(sold) / prev_size if prev_size else 0.0

    cost_lookup = (
        sub.assign(ticker=sub["ticker"].astype(str))
        .set_index("ticker")["transaction_cost_bps"]
        .to_dict()
    )
    default_bps = float(pd.to_numeric(sub["transaction_cost_bps"], errors="coerce").median())
    if not np.isfinite(default_bps):
        default_bps = float(one_way_bps)

    bought_cost = sum(cost_lookup.get(ticker, default_bps) / 10000.0 for ticker in bought)
    sold_cost = len(sold) * (default_bps / 10000.0)
    cost = bought_cost / curr_size
    if prev_size:
        cost += sold_cost / prev_size
    turnover = bought_weight + sold_weight
    return float(cost), float(turnover), curr_assets


def compute_decile_portfolio_returns(
    grouped_df,
    n_groups=10,
    transaction_cost=False,
    transaction_cost_bps=1.0,
    mid_cap_transaction_cost_bps=5.0,
    small_cap_transaction_cost_bps=10.0,
):
    """
    Compute equal-weight decile returns by rebalance date.

    Returns one row per date with group_1_ret ... group_10_ret, plus
    long_top_ret, short_bottom_ret, and long_short_ret. When transaction_cost is
    True, net return, turnover, and transaction-cost columns are also included.
    """
    if grouped_df.empty:
        return pd.DataFrame()

    if transaction_cost and "transaction_cost_bps" not in grouped_df.columns:
        grouped_df = _assign_marketcap_transaction_bps(
            grouped_df,
            base_bps=transaction_cost_bps,
            mid_bps=mid_cap_transaction_cost_bps,
            small_bps=small_cap_transaction_cost_bps,
        )
    prev_assets_by_group = {group_id: None for group_id in range(1, n_groups + 1)}

    rows = []
    for date, group in grouped_df.groupby("date", sort=True):
        row = {"date": date}
        for group_id in range(1, n_groups + 1):
            sub = group[group["portfolio_group"] == group_id]
            gross_ret = float(sub["ret"].mean()) if len(sub) else np.nan
            row[f"group_{group_id}_ret"] = gross_ret
            row[f"group_{group_id}_n"] = int(len(sub))
            if transaction_cost:
                cost, turnover, curr_assets = _membership_rebalance_cost(
                    sub,
                    prev_assets_by_group[group_id],
                    transaction_cost_bps,
                )
                prev_assets_by_group[group_id] = curr_assets
                row[f"group_{group_id}_transaction_cost"] = cost
                row[f"group_{group_id}_turnover"] = turnover
                row[f"group_{group_id}_net_ret"] = (
                    gross_ret - cost if np.isfinite(gross_ret) else np.nan
                )

        row["short_bottom_ret"] = -row["group_1_ret"]
        row["long_top_ret"] = row[f"group_{n_groups}_ret"]
        row["long_short_ret"] = row["long_top_ret"] - row["group_1_ret"]
        if transaction_cost:
            row["short_bottom_net_ret"] = -row["group_1_net_ret"]
            row["long_top_net_ret"] = row[f"group_{n_groups}_net_ret"]
            row["long_short_net_ret"] = row["long_top_net_ret"] - row["group_1_net_ret"]
            row["long_short_transaction_cost"] = (
                row[f"group_{n_groups}_transaction_cost"]
                + row["group_1_transaction_cost"]
            )
            row["long_short_turnover"] = (
                row[f"group_{n_groups}_turnover"] + row["group_1_turnover"]
            )
        row["n_stocks"] = int(len(group))
        rows.append(row)

    returns = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return returns


def _max_drawdown(period_returns):
    values = np.cumprod(1.0 + np.asarray(period_returns, dtype=float))
    peaks = np.maximum.accumulate(values)
    drawdowns = values / peaks - 1.0
    return float(np.nanmin(drawdowns)) if len(drawdowns) else np.nan


def summarize_strategy_returns(
    portfolio_returns,
    return_cols=("long_top_ret", "short_bottom_ret", "long_short_ret"),
    periods_per_year=None,
    R=None,
    freq=None,
    annualization="js_cnn",
):
    """
    Summarize portfolio performance.

    Parameters
    ----------
    annualization : {"js_cnn", "compound"}
        "js_cnn" uses the same simple annualization style as JS_cnn:
        annualized_return = mean_period_return * periods_per_year
        Sharpe = annualized_return / annualized_volatility

        "compound" keeps the original Reimage behavior:
        annualized_return = (1 + mean_period_return) ** periods_per_year - 1
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(freq=freq, R=R)
    if annualization not in {"js_cnn", "compound"}:
        raise ValueError("annualization must be either 'js_cnn' or 'compound'")

    rows = []
    for col in return_cols:
        if col not in portfolio_returns.columns:
            continue
        r = pd.to_numeric(portfolio_returns[col], errors="coerce").dropna()
        if r.empty:
            continue

        mean_period = float(r.mean())
        vol_period = float(r.std(ddof=1))
        if annualization == "js_cnn":
            ann_return = float(mean_period * periods_per_year)
        else:
            ann_return = float((1.0 + mean_period) ** periods_per_year - 1.0)
        ann_vol = float(vol_period * np.sqrt(periods_per_year))
        sharpe = float(ann_return / ann_vol) if ann_vol > 0 else np.nan
        cumulative_return = float(np.prod(1.0 + r) - 1.0)
        compound_ann_return = float((1.0 + mean_period) ** periods_per_year - 1.0)
        compound_sharpe = (
            float(mean_period / vol_period * np.sqrt(periods_per_year))
            if vol_period > 0 else np.nan
        )

        rows.append({
            "strategy": col,
            "n_periods": int(len(r)),
            "periods_per_year": float(periods_per_year),
            "annualization_method": annualization,
            "mean_period_return": mean_period,
            "period_volatility": vol_period,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe": sharpe,
            "compound_annualized_return": compound_ann_return,
            "compound_sharpe": compound_sharpe,
            "cumulative_return": cumulative_return,
            "max_drawdown": _max_drawdown(r),
            "positive_period_rate": float((r > 0).mean()),
        })

    return pd.DataFrame(rows)


def run_decile_backtest(
    pred_df,
    n_groups=10,
    R=None,
    freq=None,
    periods_per_year=None,
    annualization="js_cnn",
    output_dir=None,
    prefix="test",
    transaction_cost=False,
    transaction_cost_bps=1.0,
    mid_cap_transaction_cost_bps=5.0,
    small_cap_transaction_cost_bps=10.0,
    marketcap_path=None,
):
    grouped_df = assign_decile_groups(pred_df, n_groups=n_groups)
    if transaction_cost:
        grouped_d`f = add_marketcap_from_us_week_ret(
            grouped_df,
            marketcap_path=marketcap_path,
        )
        grouped_df = _assign_marketcap_transaction_bps(
            grouped_df,
            base_bps=transaction_cost_bps,
            mid_bps=mid_cap_transaction_cost_bps,
            small_bps=small_cap_transaction_cost_bps,
        )
    portfolio_returns = compute_decile_portfolio_returns(
        grouped_df,
        n_groups=n_groups,
        transaction_cost=transaction_cost,
        transaction_cost_bps=transaction_cost_bps,
        mid_cap_transaction_cost_bps=mid_cap_transaction_cost_bps,
        small_cap_transaction_cost_bps=small_cap_transaction_cost_bps,
    )
    return_cols = ("long_top_ret", "short_bottom_ret", "long_short_ret")
    if transaction_cost:
        return_cols = return_cols + (
            "long_top_net_ret",
            "short_bottom_net_ret",
            "long_short_net_ret",
        )
    summary = summarize_strategy_returns(
        portfolio_returns,
        return_cols=return_cols,
        periods_per_year=periods_per_year,
        R=R,
        freq=freq,
        annualization=annualization,
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        grouped_df.to_csv(
            os.path.join(output_dir, f"{prefix}_portfolio_assignments.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        portfolio_returns.to_csv(
            os.path.join(output_dir, f"{prefix}_portfolio_returns.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        summary.to_csv(
            os.path.join(output_dir, f"{prefix}_portfolio_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    return grouped_df, portfolio_returns, summary
