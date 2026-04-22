from __future__ import annotations

import numpy as np
import pandas as pd


# Jiang, Kelly, Xiu style price-image geometry.
BAR_WIDTH = 3
VOLUME_CHART_GAP = 1
IMAGE_HEIGHTS = {5: 32, 20: 64, 60: 96}


def draw_line(
    image: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    value: float = 255.0,
):
    """Draw a one-pixel line with linear interpolation."""
    steps = max(abs(r1 - r0), abs(c1 - c0)) + 1
    rr = np.rint(np.linspace(r0, r1, steps)).astype(int)
    cc = np.rint(np.linspace(c0, c1, steps)).astype(int)
    valid = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
    image[rr[valid], cc[valid]] = value


def _to_float_array(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _adjust_ohlc_with_returns(window_df: pd.DataFrame):
    raw_open = _to_float_array(window_df, "open")
    raw_high = _to_float_array(window_df, "high")
    raw_low = _to_float_array(window_df, "low")
    raw_close = _to_float_array(window_df, "close")

    if "ret" in window_df.columns:
        rets = pd.to_numeric(window_df["ret"], errors="coerce")
    else:
        rets = pd.Series(raw_close).pct_change(fill_method=None)

    if not (np.isfinite(raw_close[0]) and raw_close[0] != 0):
        raise ValueError("first close is missing or zero")

    close_path = np.full(len(window_df), np.nan, dtype=float)
    close_path[0] = 1.0
    for t in range(1, len(window_df)):
        if pd.isna(rets.iloc[t]):
            continue
        previous_valid = np.where(np.isfinite(close_path[:t]))[0]
        prev_close = close_path[previous_valid[-1]] if len(previous_valid) else 1.0
        close_path[t] = prev_close * (1.0 + float(rets.iloc[t]))

    close_scale = np.divide(
        close_path,
        raw_close,
        out=np.full(len(window_df), np.nan, dtype=float),
        where=np.isfinite(close_path) & np.isfinite(raw_close) & (raw_close != 0),
    )
    scale = pd.Series(close_scale).ffill().to_numpy()

    open_path = raw_open * scale
    high_path = raw_high * scale
    low_path = raw_low * scale
    close_tick_path = np.where(np.isfinite(raw_close), close_path, np.nan)
    return open_path, high_path, low_path, close_tick_path


def generate_research_image(
    window_df: pd.DataFrame,
    I: int,
    has_volume_bar: bool = True,
    ma_lags: list[int] | None = None,
) -> np.ndarray:
    """
    Generate an OHLC + optional MA + volume image.

    `window_df` may include pre-window history. Only the last `I` rows are drawn,
    while MA lines are computed from all supplied rows, matching the paper code's
    "load I + ma_lag rows, draw the last I rows" behavior.
    """
    if I not in IMAGE_HEIGHTS:
        raise ValueError("I must be one of 5, 20, or 60")
    if len(window_df) < I:
        raise ValueError("window_df must contain at least I rows")
    if ma_lags is None:
        ma_lags = [] if I == 5 else [I]

    img_height = IMAGE_HEIGHTS[I]
    img_width = I * BAR_WIDTH
    vol_area_h = img_height // 5 if has_volume_bar else 0
    gap_h = VOLUME_CHART_GAP if has_volume_bar else 0
    price_area_h = img_height - vol_area_h - gap_h
    if price_area_h <= 1:
        raise ValueError("price area is too small")

    open_path, high_path, low_path, close_path = _adjust_ohlc_with_returns(window_df)
    vol = _to_float_array(window_df, "volume")

    chart_start = len(window_df) - I
    start_close = close_path[chart_start]
    if not (np.isfinite(start_close) and start_close != 0):
        raise ValueError("chart start close cannot be normalized")

    open_path = open_path / start_close
    high_path = high_path / start_close
    low_path = low_path / start_close
    close_path = close_path / start_close

    ma_paths = []
    for lag in ma_lags:
        lag = int(lag)
        if lag <= 0 or len(window_df) < I + lag:
            continue
        ma_paths.append(
            pd.Series(close_path).rolling(window=lag, min_periods=lag).mean().to_numpy()
        )

    open_draw = open_path[chart_start:]
    high_draw = high_path[chart_start:]
    low_draw = low_path[chart_start:]
    close_draw = close_path[chart_start:]
    vol_draw = vol[chart_start:]
    ma_draws = [ma_path[chart_start:] for ma_path in ma_paths]

    price_parts = [open_draw, high_draw, low_draw, close_draw]
    price_parts.extend(ma_draws)
    all_prices = np.concatenate(price_parts)
    if np.all(np.isnan(all_prices)):
        raise ValueError("price range cannot be computed for this window")

    p_min = np.nanmin(all_prices)
    p_max = np.nanmax(all_prices)
    if not (np.isfinite(p_min) and np.isfinite(p_max)):
        raise ValueError("price range cannot be computed for this window")
    if p_max == p_min:
        raise ValueError("price range is flat")

    def price_to_row(price):
        scaled = (price - p_min) / (p_max - p_min) * (price_area_h - 1)
        row = int(np.rint((price_area_h - 1) - scaled))
        return min(max(row, 0), price_area_h - 1)

    v_max = np.nanmax(np.abs(vol_draw)) if has_volume_bar else np.nan
    if has_volume_bar and (not np.isfinite(v_max) or v_max <= 0):
        v_max = 1.0

    image = np.zeros((img_height, img_width), dtype=np.float32)
    ma_prev = [(None, None) for _ in ma_draws]

    for i in range(I):
        col_left = i * BAR_WIDTH
        col_mid = col_left + 1
        col_right = col_left + 2

        if np.isfinite(high_draw[i]) and np.isfinite(low_draw[i]):
            high_row = price_to_row(high_draw[i])
            low_row = price_to_row(low_draw[i])
            top = min(high_row, low_row)
            bottom = max(high_row, low_row)
            image[top : bottom + 1, col_mid] = 255.0

        if np.isfinite(open_draw[i]):
            image[price_to_row(open_draw[i]), col_left] = 255.0

        if np.isfinite(close_draw[i]):
            image[price_to_row(close_draw[i]), col_right] = 255.0

        for ma_idx, ma_path in enumerate(ma_draws):
            prev_row, prev_col = ma_prev[ma_idx]
            if np.isfinite(ma_path[i]):
                ma_row = price_to_row(ma_path[i])
                image[ma_row, col_mid] = 255.0
                if prev_row is not None:
                    draw_line(image, prev_row, prev_col, ma_row, col_mid, value=255.0)
                ma_prev[ma_idx] = (ma_row, col_mid)
            else:
                ma_prev[ma_idx] = (None, None)

        if has_volume_bar and np.isfinite(vol_draw[i]) and vol_draw[i] > 0:
            vol_height = int(np.rint(abs(vol_draw[i]) / v_max * vol_area_h))
            vol_height = min(max(vol_height, 1), vol_area_h)
            image[img_height - vol_height : img_height, col_mid] = 255.0

    return image.astype(np.uint8, copy=False)
