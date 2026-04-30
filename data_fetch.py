import io
import os
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
from tqdm import tqdm


TWSE_LISTED_COMPANY_CSV = "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv"
TPEX_OTC_COMPANY_CSV = "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"
NASDAQ_LISTED_TXT = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_TXT = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def _has_crsp_files(path: Path) -> bool:
    # 確認是否有crsp的資料
    return any(path.glob("crsp_*.parquet")) or any(path.glob("crsp_*.pq"))


def resolve_crsp_data_dir(
    crsp_data_dir: str | None = None,
    env_var: str = "CRSP_DATA_DIR",
) -> str:
    """
    Resolve the CRSP data folder across machines.

    Priority:
    1. Explicit function argument
    2. CRSP_DATA_DIR environment variable
    3. Common local folders near the notebook/repo
    """
    if crsp_data_dir:
        path = Path(crsp_data_dir).expanduser()
        if _has_crsp_files(path):
            return str(path)

    env_value = os.environ.get(env_var)
    if env_value:
        path = Path(eunv_vale).expanduser()
        if _has_crsp_files(path):
            return str(path)
        raise FileNotFoundError(
            f"{env_var}={path} does not contain files like crsp_1993.parquet."
        )

    module_dir = Path(__file__).resolve().parent
    candidates = [
        Path("us_stock_data"),
        Path("us_stock"),
        Path("../us_stock_data"),
        Path("../us_stock"),
        module_dir / "us_stock_data",
        module_dir / "us_stock",
        module_dir.parent / "us_stock_data",
        module_dir.parent / "us_stock",
    ]
    for path in candidates:
        path = path.expanduser()
        if _has_crsp_files(path):
            return str(path)

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find CRSP parquet files. Set CRSP_DATA_DIR to the folder "
        f"containing files like crsp_1993.parquet. Searched: {searched}"
    )


def _years_between(start: str, end: str) -> range:
    start_year = pd.to_datetime(start).year
    end_year = pd.to_datetime(end).year
    return range(start_year, end_year + 1)


def _find_crsp_file(crsp_data_dir: str, year: int) -> Path | None:
    data_dir = Path(crsp_data_dir)
    for suffix in (".parquet", ".pq"):
        path = data_dir / f"crsp_{year}{suffix}"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def get_crsp_permno_universe(
    crsp_data_dir: str = "us_stock",
    start: str | None = None,
    end: str | None = None,
    return_info: bool = False,
    use_cache: bool = True,
):
    """
    Get available CRSP permno identifiers from yearly parquet files.

    This does not use yfinance and is intended for US CRSP data.
    """
    data_dir = Path(crsp_data_dir)
    cache_path = None
    if use_cache and start is not None and end is not None:
        start_tag = str(start).replace("-", "")
        end_tag = str(end).replace("-", "")
        cache_path = data_dir / f"_crsp_permno_universe_{start_tag}_{end_tag}.csv"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            summary = pd.read_csv(cache_path, dtype={"ticker": str, "permno": str})
            tickers = summary["ticker"].tolist()
            return (tickers, summary) if return_info else tickers

    paths = sorted(data_dir.glob("crsp_*.parquet")) + sorted(data_dir.glob("crsp_*.pq"))
    if start is not None and end is not None:
        wanted_years = set(_years_between(start, end))
        paths = [
            path for path in paths
            if path.stem.split("_")[-1].isdigit()
            and int(path.stem.split("_")[-1]) in wanted_years
        ]

    rows = []
    for path in tqdm(paths, desc="scan CRSP permnos"):
        if path.stat().st_size == 0:
            continue
        year = int(path.stem.split("_")[-1])
        df = pd.read_parquet(path, columns=["permno"])
        counts = df["permno"].dropna().value_counts(sort=False)
        rows.extend(
            {"ticker": str(permno), "permno": str(permno), "year": year, "rows": int(count)}
            for permno, count in counts.items()
        )

    info = pd.DataFrame(rows, columns=["ticker", "permno", "year", "rows"])
    if info.empty:
        return ([], info) if return_info else []

    summary = (
        info.groupby(["ticker", "permno"], as_index=False)
        .agg(first_year=("year", "min"), last_year=("year", "max"), rows=("rows", "sum"))
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(cache_path, index=False)

    tickers = summary["ticker"].tolist()
    return (tickers, summary) if return_info else tickers


def get_crsp_trading_calendar(
    crsp_data_dir: str = "us_stock",
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DatetimeIndex:
    """Return the CRSP market dates available in the yearly parquet files."""
    data_dir = Path(crsp_data_dir)
    cache_dir = data_dir / "crsp_trading_calendar"
    cache_path = None
    if use_cache and start is not None and end is not None:
        start_tag = str(start).replace("-", "")
        end_tag = str(end).replace("-", "")
        cache_path = cache_dir / f"_crsp_trading_calendar_{start_tag}_{end_tag}.csv"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            cached = pd.read_csv(cache_path)
            return pd.DatetimeIndex(pd.to_datetime(cached["date"], errors="coerce").dropna())

    dates = []
    if start is not None and end is not None:
        paths = [
            path
            for year in _years_between(start, end)
            for path in [_find_crsp_file(crsp_data_dir, year)]
            if path is not None
        ]
    else:
        paths = sorted(data_dir.glob("crsp_*.parquet")) + sorted(data_dir.glob("crsp_*.pq"))

    for path in paths:
        if path.stat().st_size == 0:
            continue
        df_year = pd.read_parquet(path, columns=["date"])
        year_dates = pd.to_datetime(df_year["date"], errors="coerce").dropna().unique()
        dates.extend(year_dates)

    calendar = pd.DatetimeIndex(pd.to_datetime(dates)).drop_duplicates().sort_values()
    if start is not None:
        calendar = calendar[calendar >= pd.to_datetime(start)]
    if end is not None:
        calendar = calendar[calendar < pd.to_datetime(end)]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": calendar}).to_csv(cache_path, index=False)
    return calendar


def _normalize_crsp_frame(df: pd.DataFrame, adjusted: bool = True) -> pd.DataFrame:
    """Convert raw CRSP columns to the OHLCV schema used by the image pipeline."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["openprc", "askhi", "bidlo", "prc", "vol", "ret", "cfacpr", "cfacshr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    price_adjuster = df["cfacpr"].replace(0, float("nan"))
    share_adjuster = df["cfacshr"].replace(0, float("nan"))

    if adjusted:
        df["open"] = (df["openprc"].abs() / price_adjuster).astype("float32")
        df["high"] = (df["askhi"].abs() / price_adjuster).astype("float32")
        df["low"] = (df["bidlo"].abs() / price_adjuster).astype("float32")
        df["close"] = (df["prc"].abs() / price_adjuster).astype("float32")
        df["volume"] = (df["vol"] * share_adjuster).astype("float32")
    else:
        df["open"] = df["openprc"].abs().astype("float32")
        df["high"] = df["askhi"].abs().astype("float32")
        df["low"] = df["bidlo"].abs().astype("float32")
        df["close"] = df["prc"].abs().astype("float32")
        df["volume"] = df["vol"].astype("float32")

    df["ret"] = df["ret"].astype("float32")
    out = df[["permno", "date", "open", "high", "low", "close", "volume", "ret"]]
    out = out.dropna(subset=["date"])
    return out.sort_values(["permno", "date"]).reset_index(drop=True)


def load_crsp_years(
    years,
    crsp_data_dir: str = "us_stock_data",
    adjusted: bool = True,
    permnos=None,
) -> pd.DataFrame | None:
    """Load one or more yearly CRSP files and return normalized OHLCV rows."""
    columns = ["permno", "date", "openprc", "askhi", "bidlo", "prc", "vol", "ret", "cfacpr", "cfacshr"]
    frames = []
    permno_set = None
    numeric_permno_set = None
    if permnos is not None:
        permno_set = {str(p) for p in permnos}
        numeric_permnos = pd.to_numeric(pd.Series(list(permno_set)), errors="coerce").dropna()
        if len(numeric_permnos) == len(permno_set):
            numeric_permno_set = set(numeric_permnos.astype("int64").tolist())

    for year in sorted(set(int(y) for y in years)):
        path = _find_crsp_file(crsp_data_dir, year)
        if path is None:
            continue
        df_year = pd.read_parquet(path, columns=columns)
        if numeric_permno_set is not None:
            df_year = df_year[df_year["permno"].isin(numeric_permno_set)].copy()
        elif permno_set is not None:
            df_year = df_year[df_year["permno"].astype(str).isin(permno_set)].copy()
        if not df_year.empty:
            frames.append(_normalize_crsp_frame(df_year, adjusted=adjusted))

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True).sort_values(["permno", "date"]).reset_index(drop=True)


def load_crsp_single(
    permno,
    start: str,
    end: str,
    crsp_data_dir: str = "us_stock_data",
    adjusted: bool = True,
) -> pd.DataFrame | None:
    """
    Load one permno from yearly CRSP parquet files and return OHLCV columns.

    Output columns match the existing image pipeline:
    date, open, high, low, close, volume.
    """
    frames = []
    columns = ["permno", "date", "openprc", "askhi", "bidlo", "prc", "vol", "ret", "cfacpr", "cfacshr"]

    for year in _years_between(start, end):
        path = _find_crsp_file(crsp_data_dir, year)
        if path is None:
            continue
        df_year = pd.read_parquet(path, columns=columns)
        df_year = df_year[df_year["permno"].astype(str) == str(permno)].copy()
        if not df_year.empty:
            frames.append(df_year)

    if not frames:
        return None

    out = _normalize_crsp_frame(pd.concat(frames, ignore_index=True), adjusted=adjusted)
    out = out[(out["date"] >= pd.to_datetime(start)) & (out["date"] < pd.to_datetime(end))]
    out = out.drop(columns=["permno"])
    out = out.sort_values("date").reset_index(drop=True)
    return out if not out.empty else None

def get_ticker_cache_paths(
    save_dir: str,
    Market: str,
    start: str,
    end: str,
) -> tuple[str, str]:
    """Return organized paths for eligible and excluded ticker cache files."""
    ticker_dir = os.path.join(save_dir, "tickers", Market)
    eligible_path = os.path.join(ticker_dir, f"{Market}_tickers_{start}_{end}.csv")
    excluded_path = os.path.join(ticker_dir, f"{Market}_tickers_excluded_{start}_{end}.csv")
    return eligible_path, excluded_path


def _read_mops_csv(url: str) -> pd.DataFrame:
    """Read a MOPS open-data CSV with a browser-like user agent."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        content = response.read().decode("utf-8-sig")
    return pd.read_csv(io.StringIO(content), dtype=str)


def _read_nasdaq_symbol_dir(url: str) -> pd.DataFrame:
    """Read a Nasdaq Trader pipe-delimited symbol directory file."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        content = response.read().decode("utf-8-sig")

    lines = [
        line
        for line in content.splitlines()
        if "|" in line and not line.startswith("File Creation Time")
    ]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)


def _to_yahoo_us_ticker(symbol: str) -> str:
    """Convert exchange symbols to Yahoo Finance format."""
    return str(symbol).strip().replace(".", "-")


def _filter_us_common_stocks(info: pd.DataFrame) -> pd.DataFrame:
    """Remove common non-company security types from the US symbol universe."""
    filtered = info.copy()
    filtered["name_upper"] = filtered["name"].fillna("").str.upper()
    filtered["raw_symbol"] = filtered["raw_symbol"].fillna("").str.strip()

    excluded_name_pattern = (
        "WARRANT|RIGHT|UNIT|PREFERRED|PREFERENCE|PFD|"
        "NOTE|BOND|DEBENTURE|ETN|FUND|CLOSED END"
    )

    mask = (
        (filtered["is_etf"].fillna("N").str.upper() != "Y")
        & (filtered["is_test"].fillna("N").str.upper() != "Y")
        & ~filtered["raw_symbol"].str.contains(r"[$^=/]", regex=True, na=False)
        & ~filtered["name_upper"].str.contains(excluded_name_pattern, regex=True, na=False)
    )
    filtered = filtered[mask].copy()
    return filtered.drop(columns=["name_upper"])


def get_us_stock_universe(
    include_nasdaq: bool = True,
    include_nyse: bool = True,
    include_amex: bool = True,
    common_stock_only: bool = True,
    return_info: bool = False,
):
    """
    Get current US stock tickers listed on NASDAQ, NYSE, and NYSE American.

    The ticker format is converted for Yahoo Finance, e.g. BRK.B -> BRK-B.
    """
    frames = []

    if include_nasdaq:
        nasdaq = _read_nasdaq_symbol_dir(NASDAQ_LISTED_TXT)
        nasdaq_info = pd.DataFrame({
            "ticker": nasdaq["Symbol"].map(_to_yahoo_us_ticker),
            "raw_symbol": nasdaq["Symbol"].astype(str).str.strip(),
            "name": nasdaq["Security Name"].astype(str).str.strip(),
            "market": "NASDAQ",
            "exchange_code": "Q",
            "is_etf": nasdaq.get("ETF", "N"),
            "is_test": nasdaq.get("Test Issue", "N"),
        })
        frames.append(nasdaq_info)

    if include_nyse or include_amex:
        other = _read_nasdaq_symbol_dir(OTHER_LISTED_TXT)
        exchange_codes = []
        if include_nyse:
            exchange_codes.append("N")
        if include_amex:
            exchange_codes.append("A")

        other = other[other["Exchange"].isin(exchange_codes)].copy()
        market_map = {"N": "NYSE", "A": "AMEX"}
        other_info = pd.DataFrame({
            "ticker": other["ACT Symbol"].map(_to_yahoo_us_ticker),
            "raw_symbol": other["ACT Symbol"].astype(str).str.strip(),
            "name": other["Security Name"].astype(str).str.strip(),
            "market": other["Exchange"].map(market_map),
            "exchange_code": other["Exchange"],
            "is_etf": other.get("ETF", "N"),
            "is_test": other.get("Test Issue", "N"),
        })
        frames.append(other_info)

    if not frames:
        empty = pd.DataFrame(
            columns=["ticker", "raw_symbol", "name", "market", "exchange_code"]
        )
        return ([], empty) if return_info else []

    info = pd.concat(frames, ignore_index=True)
    info = info.dropna(subset=["ticker"])
    info = info[info["ticker"].astype(str).str.len() > 0].copy()

    if common_stock_only:
        info = _filter_us_common_stocks(info)

    info = info.drop_duplicates(subset=["ticker"])
    info = info.sort_values(["market", "ticker"]).reset_index(drop=True)
    info = info[["ticker", "raw_symbol", "name", "market", "exchange_code", "is_etf", "is_test"]]

    tickers = info["ticker"].tolist()
    return (tickers, info) if return_info else tickers


def get_eligible_us_tickers(
    start: str,
    end: str,
    include_nasdaq: bool = True,
    include_nyse: bool = True,
    include_amex: bool = True,
    common_stock_only: bool = True,
    start_tolerance_days: int = 10,
    end_tolerance_days: int = 10,
    min_rows: int | None = None,
    max_workers: int = 8,
):
    """
    Get US tickers, then remove IPO/inactive tickers by price coverage.
    """
    tickers, info = get_us_stock_universe(
        include_nasdaq=include_nasdaq,
        include_nyse=include_nyse,
        include_amex=include_amex,
        common_stock_only=common_stock_only,
        return_info=True,
    )
    eligible_tickers, excluded = filter_active_tickers(
        tickers=tickers,
        start=start,
        end=end,
        start_tolerance_days=start_tolerance_days,
        end_tolerance_days=end_tolerance_days,
        min_rows=min_rows,
        max_workers=max_workers,
    )
    eligible_info = info[info["ticker"].isin(eligible_tickers)].reset_index(drop=True)
    return eligible_tickers, eligible_info, excluded


def get_tw_stock_universe(
    include_listed: bool = True,
    include_otc: bool = True,
    stock_only: bool = True,
    return_info: bool = False,
):
    """
    Get current Taiwan listed/OTC company tickers for Yahoo Finance.

    Returns tickers like:
      - listed stocks: 2330.TW
      - OTC stocks: 6488.TWO

    Parameters
    ----------
    include_listed : bool
        Include TWSE listed companies.
    include_otc : bool
        Include TPEx OTC companies.
    stock_only : bool
        Keep 4-digit numeric company codes only.
    return_info : bool
        If True, return (tickers, info_df). Otherwise return tickers only.
    """
    frames = []

    if include_listed:
        listed = _read_mops_csv(TWSE_LISTED_COMPANY_CSV)
        listed["market"] = "TWSE"
        listed["yahoo_suffix"] = ".TW"
        listed["listed_date"] = listed.get("上市日期")
        frames.append(listed)

    if include_otc:
        otc = _read_mops_csv(TPEX_OTC_COMPANY_CSV)
        otc["market"] = "TPEX"
        otc["yahoo_suffix"] = ".TWO"
        otc["listed_date"] = otc.get("上櫃日期")
        frames.append(otc)

    if not frames:
        empty = pd.DataFrame(columns=["ticker", "code", "name", "market", "listed_date"])
        return ([], empty) if return_info else []

    info = pd.concat(frames, ignore_index=True)
    info["code"] = info["公司代號"].astype(str).str.strip()
    info["name"] = info["公司簡稱"].astype(str).str.strip()

    if stock_only:
        info = info[info["code"].str.fullmatch(r"\d{4}", na=False)].copy()

    info["ticker"] = info["code"] + info["yahoo_suffix"]
    info = info[["ticker", "code", "name", "market", "listed_date"]]
    info = info.drop_duplicates(subset=["ticker"]).sort_values(["market", "code"])
    info = info.reset_index(drop=True)

    tickers = info["ticker"].tolist()
    return (tickers, info) if return_info else tickers


def get_eligible_tw_tickers(
    start: str,
    end: str,
    include_listed: bool = True,
    include_otc: bool = True,
    stock_only: bool = True,
    start_tolerance_days: int = 10,
    end_tolerance_days: int = 10,
    min_rows: int | None = None,
    max_workers: int = 8,
):
    """
    Get Taiwan market tickers, then remove IPO/inactive tickers by price coverage.

    This combines get_tw_stock_universe() and filter_active_tickers().
    Use it when you want a clean TICKERS list before building images.
    """
    tickers, info = get_tw_stock_universe(
        include_listed=include_listed,
        include_otc=include_otc,
        stock_only=stock_only,
        return_info=True,
    )
    eligible_tickers, excluded = filter_active_tickers(
        tickers=tickers,
        start=start,
        end=end,
        start_tolerance_days=start_tolerance_days,
        end_tolerance_days=end_tolerance_days,
        min_rows=min_rows,
        max_workers=max_workers,
    )
    eligible_info = info[info["ticker"].isin(eligible_tickers)].reset_index(drop=True)
    return eligible_tickers, eligible_info, excluded


def _download_single(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Download one ticker from Yahoo Finance and normalize columns.

    The returned DataFrame uses lowercase single-level columns and includes a
    date column when Yahoo Finance provides a date/datetime index.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" not in df.columns:
        for c in df.columns:
            if "date" in c or "time" in c:
                df = df.rename(columns={c: "date"})
                break

    return df


def _check_ticker_coverage(
    ticker: str,
    start: str,
    end: str,
    latest_allowed_start,
    earliest_allowed_end,
    min_rows: int | None,
):
    """Return (ticker, exclusion_row). exclusion_row is None when ticker passes."""
    try:
        df = _download_single(ticker, start, end)
    except Exception as exc:
        return ticker, {
            "ticker": ticker,
            "reason": f"download_error: {type(exc).__name__}",
            "first_date": pd.NaT,
            "last_date": pd.NaT,
            "rows": 0,
        }

    if df is None or df.empty or "date" not in df.columns:
        return ticker, {
            "ticker": ticker,
            "reason": "no_data",
            "first_date": pd.NaT,
            "last_date": pd.NaT,
            "rows": 0,
        }

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return ticker, {
            "ticker": ticker,
            "reason": "no_valid_dates",
            "first_date": pd.NaT,
            "last_date": pd.NaT,
            "rows": len(df),
        }

    first_date = dates.min()
    last_date = dates.max()
    reason = None

    if first_date > latest_allowed_start:
        reason = "ipo_or_listed_after_start"
    elif last_date < earliest_allowed_end:
        reason = "delisted_or_inactive_before_end"
    elif min_rows is not None and len(df) < min_rows:
        reason = "too_few_rows"

    if reason is None:
        return ticker, None

    return ticker, {
        "ticker": ticker,
        "reason": reason,
        "first_date": first_date.date(),
        "last_date": last_date.date(),
        "rows": len(df),
    }


def filter_active_tickers(
    tickers,
    start: str,
    end: str,
    start_tolerance_days: int = 10,
    end_tolerance_days: int = 10,
    min_rows: int | None = None,
    max_workers: int = 8,
):
    """
    Filter out tickers that do not cover the requested sample period.

    This removes practical IPO/new listing cases and delisted/inactive cases by
    checking whether price data spans the requested start/end range.
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    latest_allowed_start = start_dt + pd.Timedelta(days=start_tolerance_days)
    earliest_allowed_end = end_dt - pd.Timedelta(days=end_tolerance_days)

    eligible_tickers = []
    excluded_rows = []
    results = {}

    if max_workers is None or max_workers <= 1:
        for ticker in tqdm(tickers, desc="filter tickers"):
            result_ticker, excluded_row = _check_ticker_coverage(
                    ticker,
                    start,
                    end,
                    latest_allowed_start,
                    earliest_allowed_end,
                    min_rows,
                )
            results[result_ticker] = excluded_row
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _check_ticker_coverage,
                    ticker,
                    start,
                    end,
                    latest_allowed_start,
                    earliest_allowed_end,
                    min_rows,
                ): ticker
                for ticker in tickers
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="filter tickers"):
                result_ticker, excluded_row = future.result()
                results[result_ticker] = excluded_row

    for ticker in tickers:
        excluded_row = results.get(ticker)
        if excluded_row is None:
            eligible_tickers.append(ticker)
        else:
            excluded_rows.append(excluded_row)

    excluded = pd.DataFrame(
        excluded_rows,
        columns=["ticker", "reason", "first_date", "last_date", "rows"],
    )
    return eligible_tickers, excluded
