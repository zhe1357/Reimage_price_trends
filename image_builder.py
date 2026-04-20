import numpy as np
import pandas as pd


# Image dimensions used by Jiang, Kelly, Xiu style price images.
IMAGE_HEIGHTS = {5: 32, 20: 64, 60: 96}


def draw_line(image: np.ndarray, r0: int, c0: int, r1: int, c1: int, value: float = 255.0):
    """
    以線性插值在兩點之間畫線（對應論文 'connect those dots' 的 MA 連線邏輯）。
    """
    steps = max(abs(r1 - r0), abs(c1 - c0)) + 1
    rr = np.rint(np.linspace(r0, r1, steps)).astype(int)
    cc = np.rint(np.linspace(c0, c1, steps)).astype(int)
    valid = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
    image[rr[valid], cc[valid]] = value


# =========================================================
# 2. 圖像生成（對齊論文 Section I）
# =========================================================
def generate_research_image(window_df: pd.DataFrame, I: int) -> np.ndarray:
    """
    Generate an OHLC + optional MA + volume image.

    Missing fields are handled at the visual-element level:
    - missing high or low: skip the main high-low bar for that day
    - missing open: skip only the open tick
    - missing close: skip only the close tick
    - missing volume: skip only the volume bar
    - missing MA point: skip that MA point/segment
    """
    img_height = IMAGE_HEIGHTS.get(I, 64)
    img_width = I * 3

    vol_area_h = img_height // 5
    price_area_h = img_height - vol_area_h

    raw_open = pd.to_numeric(window_df["open"], errors="coerce").to_numpy(dtype=float)
    raw_high = pd.to_numeric(window_df["high"], errors="coerce").to_numpy(dtype=float)
    raw_low = pd.to_numeric(window_df["low"], errors="coerce").to_numpy(dtype=float)
    raw_close = pd.to_numeric(window_df["close"], errors="coerce").to_numpy(dtype=float)
    vol = pd.to_numeric(window_df["volume"], errors="coerce").to_numpy(dtype=float)

    if "ret" in window_df.columns:
        rets = pd.to_numeric(window_df["ret"], errors="coerce")
    else:
        px = pd.Series(raw_close)
        rets = px.pct_change(fill_method=None)

    close_path = np.full(I, np.nan, dtype=float)
    if np.isfinite(raw_close[0]) and raw_close[0] != 0:
        close_path[0] = 1.0
    for t in range(1, I):
        if pd.isna(rets.iloc[t]):
            continue
        previous_valid = np.where(np.isfinite(close_path[:t]))[0]
        prev_close = close_path[previous_valid[-1]] if len(previous_valid) else 1.0
        close_path[t] = prev_close * (1.0 + float(rets.iloc[t]))

    close_scale = np.divide(
        close_path,
        raw_close,
        out=np.full(I, np.nan, dtype=float),
        where=np.isfinite(close_path) & np.isfinite(raw_close) & (raw_close != 0),
    )
    scale = pd.Series(close_scale).ffill().to_numpy()

    open_path = raw_open * scale
    high_path = raw_high * scale
    low_path = raw_low * scale
    close_tick_path = np.where(np.isfinite(raw_close), close_path, np.nan)

    draw_ma = I != 5
    ma_path = pd.Series(close_path).rolling(window=I, min_periods=1).mean().to_numpy()

    price_parts = [open_path, high_path, low_path, close_tick_path]
    if draw_ma:
        price_parts.append(ma_path)
    all_prices = np.concatenate(price_parts)
    if np.all(np.isnan(all_prices)):
        raise ValueError("price range cannot be computed for this window")

    p_min = np.nanmin(all_prices)
    p_max = np.nanmax(all_prices)
    if not (np.isfinite(p_min) and np.isfinite(p_max)):
        raise ValueError("price range cannot be computed for this window")
    if p_max == p_min:
        p_max = p_min + 1e-8

    def price_to_row(p):
        scaled = (p - p_min) / (p_max - p_min) * (price_area_h - 1)
        return int(np.rint((price_area_h - 1) - scaled))

    v_max = np.nanmax(vol)
    if not np.isfinite(v_max) or v_max <= 0:
        v_max = 1.0

    image = np.zeros((img_height, img_width), dtype=np.float32)
    prev_ma_row = None
    prev_ma_col = None

    for i in range(I):
        col_left = i * 3
        col_mid = col_left + 1
        col_right = col_left + 2

        has_high_low = np.isfinite(high_path[i]) and np.isfinite(low_path[i])
        has_open = np.isfinite(open_path[i])
        has_close = np.isfinite(close_tick_path[i])
        has_volume = np.isfinite(vol[i]) and vol[i] > 0

        if has_high_low:
            high_row = price_to_row(high_path[i])
            low_row = price_to_row(low_path[i])
            top = min(high_row, low_row)
            bot = max(high_row, low_row)
            image[top : bot + 1, col_mid] = 255.0

        if has_open:
            open_row = price_to_row(open_path[i])
            image[open_row, col_left] = 255.0

        if has_close:
            close_row = price_to_row(close_tick_path[i])
            image[close_row, col_right] = 255.0

        if draw_ma:
            if np.isfinite(ma_path[i]):
                ma_row = price_to_row(ma_path[i])
                image[ma_row, col_mid] = 255.0
                if prev_ma_row is not None:
                    draw_line(image, prev_ma_row, prev_ma_col, ma_row, col_mid, value=255.0)
                prev_ma_row = ma_row
                prev_ma_col = col_mid
            else:
                prev_ma_row = None
                prev_ma_col = None

        if has_volume:
            vh = int(np.rint(vol[i] / v_max * (vol_area_h - 1)))
            vh = max(vh, 1)
            image[img_height - vh : img_height, col_mid] = 255.0

    return image
