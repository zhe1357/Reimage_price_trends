import os

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


def compute_decile_portfolio_returns(grouped_df, n_groups=10):
    """
    Compute equal-weight decile returns by rebalance date.

    Returns one row per date with group_1_ret ... group_10_ret, plus
    long_top_ret, short_bottom_ret, and long_short_ret.
    """
    if grouped_df.empty:
        return pd.DataFrame()

    rows = []
    for date, group in grouped_df.groupby("date", sort=True):
        row = {"date": date}
        for group_id in range(1, n_groups + 1):
            sub = group[group["portfolio_group"] == group_id]
            row[f"group_{group_id}_ret"] = float(sub["ret"].mean()) if len(sub) else np.nan
            row[f"group_{group_id}_n"] = int(len(sub))

        row["short_bottom_ret"] = -row["group_1_ret"]
        row["long_top_ret"] = row[f"group_{n_groups}_ret"]
        row["long_short_ret"] = row["long_top_ret"] - row["group_1_ret"]
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
):
    """
    Run a 10-group prediction-sorted portfolio backtest.

    The strategy:
    - each prediction date, sort stocks by pred_prob
    - split into n_groups equal-count groups
    - long the highest-probability group
    - short the lowest-probability group
    - use equal-weight future R-day returns from the `ret` column

    Returns
    -------
    grouped_df, portfolio_returns, summary
    """
    grouped_df = assign_decile_groups(pred_df, n_groups=n_groups)
    portfolio_returns = compute_decile_portfolio_returns(grouped_df, n_groups=n_groups)
    summary = summarize_strategy_returns(
        portfolio_returns,
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
