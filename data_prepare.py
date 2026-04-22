from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data_fetch import (
    _download_single,
    get_crsp_trading_calendar,
    load_crsp_single,
    load_crsp_years,
    resolve_crsp_data_dir,
)
from image_builder import generate_research_image


def _checkpoint_paths(dataset_dir: str, process_by: str = "ticker") -> dict[str, str]:
    prefix = "_checkpoint" if process_by == "ticker" else f"_checkpoint_{process_by}"
    done_name = "_checkpoint_done_tickers.txt" if process_by == "ticker" else f"{prefix}_done.txt"
    return {
        "X": os.path.join(dataset_dir, f"{prefix}_X.npy"),
        "y": os.path.join(dataset_dir, f"{prefix}_y.npy"),
        "meta": os.path.join(dataset_dir, f"{prefix}_meta.csv"),
        "done": os.path.join(dataset_dir, done_name),
    }


def _save_build_checkpoint(
    dataset_dir,
    all_images,
    all_labels,
    all_meta,
    done_tickers,
    process_by="ticker",
    save_arrays=True,
):
    paths = _checkpoint_paths(dataset_dir, process_by=process_by)
    os.makedirs(dataset_dir, exist_ok=True)

    if save_arrays and all_images:
        _atomic_save_npy(paths["X"], np.stack(all_images).astype(np.uint8))
        _atomic_save_npy(paths["y"], np.array(all_labels, dtype=np.int64))
        _atomic_save_csv(pd.DataFrame(all_meta), paths["meta"], index=False)

    with open(paths["done"], "w", encoding="utf-8") as f:
        for ticker in sorted(done_tickers):
            f.write(f"{ticker}\n")

    print(f"  Checkpoint saved: {len(done_tickers)} tickers done, {len(all_images)} images")


def _load_build_checkpoint(dataset_dir, process_by="ticker"):
    paths = _checkpoint_paths(dataset_dir, process_by=process_by)
    done_tickers = set()
    all_images = []
    all_labels = []
    all_meta = []

    if os.path.exists(paths["done"]):
        with open(paths["done"], "r", encoding="utf-8") as f:
            done_tickers = {line.strip() for line in f if line.strip()}

    has_arrays = (
        os.path.exists(paths["X"])
        and os.path.exists(paths["y"])
        and os.path.exists(paths["meta"])
    )
    if has_arrays:
        X_checkpoint = np.load(paths["X"])
        y_checkpoint = np.load(paths["y"])
        meta_checkpoint = pd.read_csv(paths["meta"])
        all_images = [img for img in X_checkpoint]
        all_labels = y_checkpoint.astype(np.int64).tolist()
        all_meta = meta_checkpoint.to_dict("records")
    elif done_tickers:
        print(
            "  Checkpoint done file exists, but checkpoint arrays/meta are missing. "
            "Ignoring done markers and rebuilding to avoid an incomplete dataset."
        )
        done_tickers = set()

    return all_images, all_labels, all_meta, done_tickers


def _final_dataset_paths(dataset_dir: str) -> dict[str, str]:
    names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    names.extend(["X_trainval", "y_trainval"])
    paths = {name: os.path.join(dataset_dir, f"{name}.npy") for name in names}
    paths["meta"] = os.path.join(dataset_dir, "meta.csv")
    paths["meta_trainval"] = os.path.join(dataset_dir, "meta_trainval.csv")
    paths["meta_test"] = os.path.join(dataset_dir, "meta_test.csv")
    return paths


def _final_dataset_exists(dataset_dir: str) -> bool:
    required = [
        "X_trainval",
        "y_trainval",
        "X_test",
        "y_test",
        "meta",
        "meta_trainval",
        "meta_test",
    ]
    paths = _final_dataset_paths(dataset_dir)
    if not all(os.path.exists(paths[name]) for name in required):
        return False
    try:
        X_tv = np.load(paths["X_trainval"], mmap_mode="r")
        y_tv = np.load(paths["y_trainval"], mmap_mode="r")
        X_te = np.load(paths["X_test"], mmap_mode="r")
        y_te = np.load(paths["y_test"], mmap_mode="r")
        meta_trainval = pd.read_csv(paths["meta_trainval"])
        meta_test = pd.read_csv(paths["meta_test"])
    except Exception as exc:
        print(f"Existing final dataset is not readable ({type(exc).__name__}: {exc})")
        return False
    return (
        len(X_tv) == len(y_tv) == len(meta_trainval)
        and len(X_te) == len(y_te) == len(meta_test)
    )


def _load_final_dataset(dataset_dir: str):
    paths = _final_dataset_paths(dataset_dir)
    X_tv = np.load(paths["X_trainval"])
    y_tv = np.load(paths["y_trainval"])
    X_te = np.load(paths["X_test"])
    y_te = np.load(paths["y_test"])
    meta_df = pd.read_csv(paths["meta"])
    return X_tv, y_tv, X_te, y_te, meta_df


def _save_feather_if_available(df: pd.DataFrame, path: str):
    try:
        tmp_path = f"{path}.tmp"
        df.to_feather(tmp_path)
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print(f"  Feather not saved ({type(exc).__name__}: {exc})")


def _safe_rate(y: np.ndarray) -> float:
    return float(np.mean(y)) if len(y) else float("nan")


def _atomic_save_npy(path: str, arr: np.ndarray):
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, arr)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            raise


def _atomic_save_csv(df: pd.DataFrame, path: str, **kwargs):
    tmp_path = f"{path}.tmp"
    try:
        df.to_csv(tmp_path, **kwargs)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            raise


def _write_build_log(
    dataset_dir,
    Market,
    start,
    end,
    train_end,
    I,
    R,
    sample_step,
    sample_mode,
    sample_freq,
    price_source,
    strict_time_split,
    max_workers,
    checkpoint_every,
    checkpoint_save_arrays,
    year_ticker_chunk_size,
    process_by,
    total_tickers,
    done_tickers,
    skipped_rows,
    dropped,
    X_tv,
    y_tv,
    X_te,
    y_te,
):
    skipped = pd.DataFrame(skipped_rows, columns=["ticker", "reason"])
    skip_counts = skipped["reason"].value_counts().to_dict() if not skipped.empty else {}
    lines = [
        f"Market: {Market}",
        f"start: {start}",
        f"end: {end}",
        f"train_end: {train_end}",
        f"I: {I}",
        f"R: {R}",
        f"sample_step: {sample_step}",
        f"sample_mode: {sample_mode}",
        f"sample_freq: {sample_freq}",
        f"price_source: {price_source}",
        f"strict_time_split: {strict_time_split}",
        f"max_workers: {max_workers}",
        f"checkpoint_every: {checkpoint_every}",
        f"checkpoint_save_arrays: {checkpoint_save_arrays}",
        f"year_ticker_chunk_size: {year_ticker_chunk_size}",
        f"process_by: {process_by}",
        f"input_tickers: {total_tickers}",
        f"done_tickers: {len(done_tickers)}",
        f"skipped_tickers: {len(skipped_rows)}",
        f"boundary_overlap_dropped_samples: {dropped}",
        f"trainval_samples: {len(X_tv)}",
        f"test_samples: {len(X_te)}",
        f"trainval_up_rate: {_safe_rate(y_tv):.6f}",
        f"test_up_rate: {_safe_rate(y_te):.6f}",
        f"skip_reason_counts: {skip_counts}",
    ]
    with open(os.path.join(dataset_dir, "build_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if skipped_rows:
        skipped.to_csv(os.path.join(dataset_dir, "skipped_tickers.csv"), index=False)


def _save_sample_images(dataset_dir, X, meta_df, limit=8):
    if limit is None or limit <= 0 or len(X) == 0:
        return

    try:
        from PIL import Image
    except Exception as exc:
        print(f"  Sample images not saved ({type(exc).__name__}: {exc})")
        return

    sample_dir = os.path.join(dataset_dir, "sample_images")
    os.makedirs(sample_dir, exist_ok=True)
    n = min(int(limit), len(X))

    for idx in range(n):
        row = meta_df.iloc[idx]
        ticker = str(row.get("ticker", "unknown")).replace(os.sep, "_")
        date = pd.to_datetime(row.get("date")).strftime("%Y-%m-%d")
        label = row.get("label", "na")
        filename = f"{idx:04d}_{ticker}_{date}_label{label}.png"
        img = np.asarray(X[idx])
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(os.path.join(sample_dir, filename))


def get_dataset_output_dir(
    save_dir: str,
    Market: str,
    I: int,
    R: int,
    sample_step: int | None = None,
    sample_mode: str | None = None,
    sample_freq: str | None = None,
) -> str:
    """Return the organized output folder for one generated dataset."""
    if sample_step is None:
        sample_step = R
    tag = f"I{I}R{R}S{sample_step}"
    if sample_mode is not None:
        tag = f"{tag}_{sample_mode}"
    if sample_freq is not None:
        tag = f"{tag}_{sample_freq}"
    return os.path.join(save_dir, "training_data", Market, tag)


def _forward_return(df: pd.DataFrame, R: int, price_source: str) -> pd.Series:
    """Compute future R-day return for labels."""
    if price_source == "crsp" and "ret" in df.columns:
        gross = 1.0 + pd.to_numeric(df["ret"], errors="coerce")
        future_gross = pd.Series(1.0, index=df.index, dtype=float)
        for k in range(1, R + 1):
            future_gross = future_gross * gross.shift(-k)
        return future_gross - 1.0

    return df["close"].pct_change(R).shift(-R)


def _infer_sample_freq(R: int) -> str:
    if R == 5:
        return "week"
    if R == 20:
        return "month"
    if R in {60, 65}:
        return "quarter"
    raise ValueError("sample_freq must be provided when R is not 5, 20, 60, or 65")


def _period_end_dates(dates: pd.Series, sample_freq: str) -> pd.DatetimeIndex:
    """Return last available trading date in each week/month/quarter."""
    if sample_freq not in {"week", "month", "quarter"}:
        raise ValueError("sample_freq must be 'week', 'month', or 'quarter'")

    date_index = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce").dropna())
    date_index = date_index.drop_duplicates().sort_values()
    if len(date_index) == 0:
        return pd.DatetimeIndex([])

    if sample_freq == "week":
        period_key = date_index.to_period("W-FRI")
    elif sample_freq == "month":
        period_key = date_index.to_period("M")
    else:
        period_key = date_index.to_period("Q")

    return pd.DatetimeIndex(pd.Series(date_index).groupby(period_key).max().to_numpy())


def _candidate_end_indices(
    df: pd.DataFrame,
    I: int,
    sample_step: int,
    sample_mode: str,
    sample_freq: str,
) -> list[int]:
    if sample_mode == "step":
        return [i + I - 1 for i in range(0, len(df) - I + 1, sample_step)]
    if sample_mode != "period_end":
        raise ValueError("sample_mode must be 'period_end' or 'step'")

    period_ends = set(_period_end_dates(df["date"], sample_freq))
    dates = pd.to_datetime(df["date"], errors="coerce")
    return [
        idx for idx, date in enumerate(dates)
        if idx >= I - 1 and pd.notna(date) and date in period_ends
    ]


def _build_ticker_samples_from_df(
    ticker,
    df,
    I,
    R,
    sample_step,
    sample_mode,
    sample_freq,
    price_source,
    trading_calendar=None,
    show_progress=False,
    target_start=None,
    target_end=None,
    has_volume_bar=True,
    ma_lags=None,
):
    ticker_key = str(ticker)
    images = []
    labels = []
    meta = []

    if df is None:
        return ticker_key, images, labels, meta, "no data"

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        return ticker_key, images, labels, meta, f"missing columns: {missing}"

    if trading_calendar is not None:
        calendar = pd.DatetimeIndex(trading_calendar)
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = (
            df.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")
            .reindex(calendar)
            .rename_axis("date")
            .reset_index()
        )

    df[f"ret{R}"] = _forward_return(df, R, price_source)
    df["label"] = (df[f"ret{R}"] > 0).where(df[f"ret{R}"].notna(), np.nan)
    valid_price_row = df[["open", "high", "low", "close", "volume"]].notna().any(axis=1)
    valid_ret_row = df[f"ret{R}"].notna()
    df = df.reset_index(drop=True)
    valid_price_row = valid_price_row.reset_index(drop=True)
    valid_ret_row = valid_ret_row.reset_index(drop=True)
    valid_indices = np.flatnonzero(valid_price_row.to_numpy())

    if len(valid_indices) == 0:
        return ticker_key, images, labels, meta, "no valid price rows"

    first_valid_idx = int(valid_indices[0])
    last_valid_idx = int(valid_indices[-1])

    if len(df) < I + R + 5:
        return ticker_key, images, labels, meta, "not enough rows"

    target_start = pd.to_datetime(target_start) if target_start is not None else None
    target_end = pd.to_datetime(target_end) if target_end is not None else None
    candidate_end_indices = _candidate_end_indices(
        df,
        I=I,
        sample_step=sample_step,
        sample_mode=sample_mode,
        sample_freq=sample_freq,
    )
    iterator = candidate_end_indices
    if show_progress:
        iterator = tqdm(iterator, desc=f"  build {ticker}")

    ma_offset = 0 if ma_lags is None else max([int(lag) for lag in ma_lags], default=0)
    for end_idx in iterator:
        i = end_idx - I + 1
        history_start = max(first_valid_idx, i - ma_offset)
        window = df.iloc[history_start : end_idx + 1]

        if i <= first_valid_idx or end_idx >= last_valid_idx:
            continue
        if not valid_ret_row.iloc[end_idx]:
            continue
        if end_idx + R >= len(df):
            continue

        end_date = df.iloc[end_idx]["date"]
        if target_start is not None and end_date < target_start:
            continue
        if target_end is not None and end_date >= target_end:
            continue

        label = int(df.iloc[end_idx]["label"])
        start_date = df.iloc[i]["date"]
        label_end_date = df.iloc[end_idx + R]["date"]

        try:
            img = generate_research_image(
                window,
                I=I,
                has_volume_bar=has_volume_bar,
                ma_lags=ma_lags,
            )
        except ValueError:
            continue

        images.append(img)
        labels.append(label)
        meta.append({
            "ticker": ticker,
            "start_date": start_date,
            "date": end_date,
            "label_end_date": label_end_date,
            "label": label,
            "ret": df.iloc[end_idx][f"ret{R}"],
            "I": I,
            "R": R,
            "sample_step": sample_step,
            "sample_mode": sample_mode,
            "sample_freq": sample_freq,
            "price_source": price_source,
        })

    return ticker_key, images, labels, meta, None


def _build_one_ticker_samples(
    ticker,
    start,
    end,
    I,
    R,
    sample_step,
    sample_mode,
    sample_freq,
    price_source,
    crsp_data_dir,
    crsp_adjusted,
    trading_calendar=None,
    show_progress=False,
    has_volume_bar=True,
    ma_lags=None,
):
    ticker_key = str(ticker)
    images = []
    labels = []
    meta = []

    if price_source == "yfinance":
        df = _download_single(ticker, start, end)
    elif price_source == "crsp":
        df = load_crsp_single(
            ticker,
            start=start,
            end=end,
            crsp_data_dir=crsp_data_dir,
            adjusted=crsp_adjusted,
        )
    else:
        raise ValueError("price_source must be 'yfinance' or 'crsp'")

    return _build_ticker_samples_from_df(
        ticker=ticker,
        df=df,
        I=I,
        R=R,
        sample_step=sample_step,
        sample_mode=sample_mode,
        sample_freq=sample_freq,
        price_source=price_source,
        trading_calendar=trading_calendar,
        show_progress=show_progress,
        has_volume_bar=has_volume_bar,
        ma_lags=ma_lags,
    )


def _year_range_for_build(start, end):
    start_year = pd.to_datetime(start).year
    end_year = pd.to_datetime(end).year
    return range(start_year, end_year + 1)


def _build_crsp_year_samples(
    year,
    tickers,
    start,
    end,
    I,
    R,
    sample_step,
    sample_mode,
    sample_freq,
    crsp_data_dir,
    crsp_adjusted,
    ticker_chunk_size=1000,
    has_volume_bar=True,
    ma_lags=None,
):
    year = int(year)
    unit_key = f"year:{year}"
    images = []
    labels = []
    meta = []
    skipped_rows = []

    global_start = pd.to_datetime(start)
    global_end = pd.to_datetime(end)
    target_start = max(pd.Timestamp(year=year, month=1, day=1), global_start)
    target_end = min(pd.Timestamp(year=year + 1, month=1, day=1), global_end)
    if target_start >= target_end:
        return unit_key, images, labels, meta, skipped_rows, "outside requested period"

    extended_start = f"{year - 1}-01-01"
    extended_end = f"{year + 2}-01-01"
    trading_calendar = get_crsp_trading_calendar(
        crsp_data_dir=crsp_data_dir,
        start=extended_start,
        end=extended_end,
    )

    years_to_read = [year - 1, year, year + 1]
    ticker_list = list(tickers)
    if ticker_chunk_size is None or ticker_chunk_size <= 0:
        ticker_chunks = [ticker_list]
    else:
        ticker_chunks = [
            ticker_list[i : i + int(ticker_chunk_size)]
            for i in range(0, len(ticker_list), int(ticker_chunk_size))
        ]

    wanted = {str(ticker) for ticker in tickers}
    seen = set()
    for chunk in tqdm(ticker_chunks, desc=f"year {year} chunks"):
        df_all = load_crsp_years(
            years=years_to_read,
            crsp_data_dir=crsp_data_dir,
            adjusted=crsp_adjusted,
            permnos=chunk,
        )
        if df_all is None or df_all.empty:
            continue

        df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
        df_all = df_all[
            (df_all["date"] >= pd.to_datetime(extended_start))
            & (df_all["date"] < pd.to_datetime(extended_end))
        ].copy()
        if df_all.empty:
            continue

        grouped = df_all.groupby("permno", sort=False)
        for permno, df_one in grouped:
            seen.add(str(permno))
            ticker_key, ticker_images, ticker_labels, ticker_meta, warning = _build_ticker_samples_from_df(
                ticker=str(permno),
                df=df_one.drop(columns=["permno"], errors="ignore"),
                I=I,
                R=R,
                sample_step=sample_step,
                sample_mode=sample_mode,
                sample_freq=sample_freq,
                price_source="crsp",
                trading_calendar=trading_calendar,
                show_progress=False,
                target_start=target_start,
                target_end=target_end,
                has_volume_bar=has_volume_bar,
                ma_lags=ma_lags,
            )
            if warning:
                skipped_rows.append({"ticker": ticker_key, "reason": f"{year}: {warning}"})
                continue
            images.extend(ticker_images)
            labels.extend(ticker_labels)
            meta.extend(ticker_meta)

    for missing_ticker in sorted(wanted - seen):
        skipped_rows.append({"ticker": missing_ticker, "reason": f"{year}: no data"})

    return unit_key, images, labels, meta, skipped_rows, None


def build_research_dataset(
    tickers,
    start,
    end,
    I    = 20,   # 圖像天數（5 / 20 / 60）
    R    = 20,   # 預測未來報酬天數（5 / 20 / 60）
    train_end  = "2020-01-01",   # 訓練+驗證截止日（test 從此日開始）
    val_ratio  = 0.3,            # 驗證集比例（論文：30%，隨機抽取）
    save_dir   = "data",
    random_state = 42,
    Market = "TW",
    sample_step = None,
    sample_mode = "period_end",
    sample_freq = None,
    price_source = "yfinance",
    crsp_data_dir = "us_stock_data",
    crsp_adjusted = True,
    has_volume_bar = True,
    ma_lags = None,
    strict_time_split = True,
    checkpoint_every = 0,
    resume = False,
    max_workers = 1,
    load_existing = False,
    sample_image_count = 8,
    process_by = "ticker",
    checkpoint_save_arrays = True,
    year_ticker_chunk_size = 1000,
):
    """
    建構符合論文設計的訓練/驗證/測試資料集。

    關鍵設計說明：
    ─────────────
    [1] auto_adjust=True：所有 OHLC 皆為 adjusted prices，
        確保 O/H/L 與 close 的比例計算不受股息/拆股影響。

    [2] 標籤：y = 1 若未來 R 天報酬 > 0，否則 y = 0。
        對齊方式：圖像結束於第 i+I-1 天，標籤為第 i+I 天
        開始的 R 天報酬方向（forward-looking）。

    [3] 訓練/驗證切割：
        - 先以 train_end 日期分出 train_val 與 test
        - train_val 內部【隨機】切割 (1-val_ratio) / val_ratio
        - 論文明確指出隨機切割的目的：讓各子集的 up/down 比例
          各接近 50%，避免因多頭/空頭市場集中造成標籤不平衡。
          （按時間切割會讓驗證集全部落在單一市場環境，
           分類器看起來只會猜多數類，準確度停在 50~60%。）

    Returns
    -------
    X_trainval, y_trainval, X_test, y_test, meta_df
    """
    if sample_step is None:
        sample_step = R
    if sample_freq is None:
        sample_freq = _infer_sample_freq(R)
    if ma_lags is None:
        ma_lags = [] if I == 5 else [I]
    if sample_step < 1:
        raise ValueError("sample_step must be >= 1")
    if sample_mode not in {"period_end", "step"}:
        raise ValueError("sample_mode must be 'period_end' or 'step'")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if process_by not in {"ticker", "year"}:
        raise ValueError("process_by must be 'ticker' or 'year'")
    if process_by == "year" and price_source != "crsp":
        raise ValueError("process_by='year' is currently only supported for price_source='crsp'")
    if price_source == "crsp":
        crsp_data_dir = resolve_crsp_data_dir(crsp_data_dir)

    dataset_dir = get_dataset_output_dir(
        save_dir,
        Market,
        I,
        R,
        sample_step=sample_step,
        sample_mode=sample_mode,
        sample_freq=sample_freq,
    )
    os.makedirs(dataset_dir, exist_ok=True)

    if load_existing and _final_dataset_exists(dataset_dir):
        print(f"Found existing final dataset: {dataset_dir}")
        return _load_final_dataset(dataset_dir)

    all_images = []
    all_labels = []
    all_meta   = []
    done_tickers = set()
    skipped_rows = []

    if resume:
        all_images, all_labels, all_meta, done_tickers = _load_build_checkpoint(
            dataset_dir,
            process_by=process_by,
        )
        print(
            f"Resume checkpoint: {len(done_tickers)} tickers done, "
            f"{len(all_images)} images loaded"
        )

    trading_calendar = None
    if price_source == "crsp" and process_by != "year":
        trading_calendar = get_crsp_trading_calendar(
            crsp_data_dir=crsp_data_dir,
            start=start,
            end=end,
        )
        print(f"CRSP trading calendar: {len(trading_calendar)} dates")

    processed_since_checkpoint = 0

    def _mark_ticker_done(ticker_key):
        nonlocal processed_since_checkpoint
        done_tickers.add(ticker_key)
        processed_since_checkpoint += 1
        if checkpoint_every and processed_since_checkpoint >= checkpoint_every:
            _save_build_checkpoint(
                dataset_dir,
                all_images,
                all_labels,
                all_meta,
                done_tickers,
                process_by=process_by,
                save_arrays=checkpoint_save_arrays,
            )
            processed_since_checkpoint = 0

    if process_by == "year":
        years = list(_year_range_for_build(start, end))
        pending_years = [year for year in years if f"year:{year}" not in done_tickers]
        print(f"Year-based CRSP build: pending years={pending_years}")

        for year in pending_years:
            print(f"\n[I{I}/R{R}] build year: {year}")
            unit_key, images, labels, meta, year_skipped, warning = _build_crsp_year_samples(
                year=year,
                tickers=tickers,
                start=start,
                end=end,
                I=I,
                R=R,
                sample_step=sample_step,
                sample_mode=sample_mode,
                sample_freq=sample_freq,
                crsp_data_dir=crsp_data_dir,
                crsp_adjusted=crsp_adjusted,
                ticker_chunk_size=year_ticker_chunk_size,
                has_volume_bar=has_volume_bar,
                ma_lags=ma_lags,
            )
            if warning:
                print(f"  skip year {year}: {warning}")
            else:
                print(f"  done year {year}: {len(images)} images")

            skipped_rows.extend(year_skipped)
            all_images.extend(images)
            all_labels.extend(labels)
            all_meta.extend(meta)
            _mark_ticker_done(unit_key)

        pending_tickers = []
    elif max_workers > 1:
        pending_tickers = [ticker for ticker in tickers if str(ticker) not in done_tickers]
        print(f"Parallel build: max_workers={max_workers}, pending tickers={len(pending_tickers)}")

        if pending_tickers:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _build_one_ticker_samples,
                        ticker,
                        start,
                        end,
                        I,
                        R,
                        sample_step,
                        sample_mode,
                        sample_freq,
                        price_source,
                        crsp_data_dir,
                        crsp_adjusted,
                        trading_calendar,
                        False,
                        has_volume_bar,
                        ma_lags,
                    ): ticker
                    for ticker in pending_tickers
                }

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"tickers x{max_workers}"):
                    ticker = futures[future]
                    try:
                        ticker_key, images, labels, meta, warning = future.result()
                    except Exception as exc:
                        ticker_key = str(ticker)
                        images, labels, meta = [], [], []
                        warning = f"failed: {exc}"

                    if warning:
                        print(f"  skip {ticker}: {warning}")
                        skipped_rows.append({"ticker": ticker, "reason": warning})
                    else:
                        print(f"  done {ticker}: {len(images)} images")

                    all_images.extend(images)
                    all_labels.extend(labels)
                    all_meta.extend(meta)
                    _mark_ticker_done(ticker_key)

        pending_tickers = []
    else:
        pending_tickers = [ticker for ticker in tickers if str(ticker) not in done_tickers]
        for ticker in pending_tickers:
            print(f"\n[I{I}/R{R}] build: {ticker}")
            try:
                ticker_key, images, labels, meta, warning = _build_one_ticker_samples(
                    ticker,
                    start,
                    end,
                    I,
                    R,
                    sample_step,
                    sample_mode,
                    sample_freq,
                    price_source,
                    crsp_data_dir,
                    crsp_adjusted,
                    trading_calendar,
                    True,
                    has_volume_bar,
                    ma_lags,
                )
            except Exception as exc:
                ticker_key = str(ticker)
                images, labels, meta = [], [], []
                warning = f"failed: {exc}"

            if warning:
                print(f"  skip {ticker}: {warning}")
                skipped_rows.append({"ticker": ticker, "reason": warning})
            else:
                print(f"  done {ticker}: {len(images)} images")

            all_images.extend(images)
            all_labels.extend(labels)
            all_meta.extend(meta)
            _mark_ticker_done(ticker_key)

        pending_tickers = []

    if checkpoint_every:
        _save_build_checkpoint(
            dataset_dir,
            all_images,
            all_labels,
            all_meta,
            done_tickers,
            process_by=process_by,
            save_arrays=checkpoint_save_arrays,
        )

    if len(all_images) == 0:
        raise RuntimeError("沒有成功生成任何圖像，請檢查 tickers 與日期範圍")

    X_all   = np.stack(all_images).astype(np.uint8)
    y_all   = np.array(all_labels, dtype=np.int64)
    meta_df = pd.DataFrame(all_meta)

    print(f"\n總樣本數: {len(X_all)}")
    print(f"標籤分布: up={y_all.mean():.3f}, down={1-y_all.mean():.3f}")

    # ── 依時間分出 test（論文：out-of-sample test）──────────────────────────
    split_date = pd.to_datetime(train_end)
    meta_df["start_date"] = pd.to_datetime(meta_df["start_date"])
    meta_df["date"] = pd.to_datetime(meta_df["date"])
    meta_df["label_end_date"] = pd.to_datetime(meta_df["label_end_date"])

    if strict_time_split:
        train_val_mask = meta_df["label_end_date"] < split_date
        test_mask = meta_df["start_date"] >= split_date
        dropped = len(meta_df) - int(train_val_mask.sum()) - int(test_mask.sum())
        print(f"Strict split dropped boundary-overlap samples: {dropped}")
    else:
        test_mask = meta_df["date"] >= split_date
        train_val_mask = ~test_mask

    train_val_indices = np.flatnonzero(train_val_mask.to_numpy())
    test_indices = np.flatnonzero(test_mask.to_numpy())

    X_tv = X_all[train_val_indices];  y_tv = y_all[train_val_indices]
    X_te = X_all[test_indices];       y_te = y_all[test_indices]
    meta_test = meta_df.iloc[test_indices].reset_index(drop=True)

    print(f"  Train+Val: {len(X_tv)}  (up={y_tv.mean():.3f})")
    print(f"  Test     : {len(X_te)}  (up={y_te.mean():.3f})")

    for name, arr in [
        ("X_trainval", X_tv), ("y_trainval", y_tv),
        ("X_test", X_te), ("y_test", y_te),
    ]:
        _atomic_save_npy(os.path.join(dataset_dir, f"{name}.npy"), arr)
    _atomic_save_csv(meta_df, os.path.join(dataset_dir, "meta.csv"), index=False)
    meta_trainval = meta_df.iloc[train_val_indices].reset_index(drop=True)
    _atomic_save_csv(meta_trainval, os.path.join(dataset_dir, "meta_trainval.csv"), index=False)
    _atomic_save_csv(meta_test, os.path.join(dataset_dir, "meta_test.csv"), index=False)
    _save_feather_if_available(meta_df, os.path.join(dataset_dir, "meta.feather"))
    _save_feather_if_available(meta_trainval, os.path.join(dataset_dir, "meta_trainval.feather"))
    _save_feather_if_available(meta_test, os.path.join(dataset_dir, "meta_test.feather"))
    _save_sample_images(dataset_dir, X_all, meta_df, limit=sample_image_count)
    _write_build_log(
        dataset_dir=dataset_dir,
        Market=Market,
        start=start,
        end=end,
        train_end=train_end,
        I=I,
        R=R,
        sample_step=sample_step,
        sample_mode=sample_mode,
        sample_freq=sample_freq,
        price_source=price_source,
        strict_time_split=strict_time_split,
        max_workers=max_workers,
        checkpoint_every=checkpoint_every,
        checkpoint_save_arrays=checkpoint_save_arrays,
        year_ticker_chunk_size=year_ticker_chunk_size,
        process_by=process_by,
        total_tickers=len(tickers),
        done_tickers=done_tickers,
        skipped_rows=skipped_rows,
        dropped=dropped,
        X_tv=X_tv,
        y_tv=y_tv,
        X_te=X_te,
        y_te=y_te,
    )

    print(f"\n撌脣摮 {dataset_dir}/")
    return X_tv, y_tv, X_te, y_te, meta_df


# =========================================================
# 4. 像素正規化（論文腳注 8）
#    只用訓練集 pixel 計算 mean/std，再套用到 val/test
# =========================================================
def fit_pixel_norm(X_train: np.ndarray):
    """用訓練集計算 pixel-level mean / std。"""
    X_float = X_train.astype(np.float32, copy=False)
    mean = float(np.mean(X_float))
    std  = float(np.std(X_float))
    if std < 1e-7:
        std = 1.0
    return mean, std


def apply_pixel_norm(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    """套用像素正規化（(X - mean) / std）。"""
    X_float = X.astype(np.float32, copy=False)
    return ((X_float - mean) / (std + 1e-7)).astype(np.float32, copy=False)


# =========================================================
# 5. PyTorch DataLoader 範例
# =========================================================
def make_dataloaders(X_tr, y_tr, X_va, y_va, X_te, y_te,
                     batch_size=128, num_workers=0):
    """
    建立 PyTorch DataLoader，自動套用像素正規化。

    CNN 輸入格式：(N, 1, H, W)（單通道灰階）
    """
    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        raise ImportError("請先安裝 PyTorch：pip install torch")

    # 像素正規化（只從訓練集 fit）
    mean, std = fit_pixel_norm(X_tr)
    X_tr_n = apply_pixel_norm(X_tr, mean, std)
    X_va_n = apply_pixel_norm(X_va, mean, std)
    X_te_n = apply_pixel_norm(X_te, mean, std)

    # 加入 channel 維度 (N, H, W) → (N, 1, H, W)
    def to_tensor(X, y):
        return TensorDataset(
            torch.from_numpy(X[:, np.newaxis, :, :]),
            torch.from_numpy(y),
        )

    train_ds = to_tensor(X_tr_n, y_tr)
    val_ds   = to_tensor(X_va_n, y_va)
    test_ds  = to_tensor(X_te_n, y_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def make_dataloaders_from_trainval(
    X_trainval,
    y_trainval,
    X_test,
    y_test,
    val_ratio=0.3,
    random_state=42,
    batch_size=128,
    num_workers=0,
):
    """Split trainval into train/val at training time, then build DataLoaders."""
    indices = np.arange(len(X_trainval))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=y_trainval,
    )

    X_tr = X_trainval[train_indices]
    y_tr = y_trainval[train_indices]
    X_va = X_trainval[val_indices]
    y_va = y_trainval[val_indices]

    return make_dataloaders(
        X_tr,
        y_tr,
        X_va,
        y_va,
        X_test,
        y_test,
        batch_size=batch_size,
        num_workers=num_workers,
    )


# =========================================================
# 6. Demo
# =========================================================
