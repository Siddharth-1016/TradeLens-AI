"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 3 — FEATURE ENGINEERING PIPELINE                      ║
║  Technical indicators + fundamental ratios + cleaning       ║
╚══════════════════════════════════════════════════════════════╝

WHY EACH FEATURE:
──────────────────────────────────────────────────────────────
SMA 20/50/200   → Trend direction at 3 horizons (short/mid/long)
EMA 12/26       → Exponential (recency-weighted) trend
RSI             → Overbought/oversold (momentum oscillator)
MACD            → Trend + momentum convergence/divergence
Bollinger %B    → Price position within volatility envelope
ATR             → Absolute volatility (risk sizing)
Volatility 20d  → Rolling standard deviation of returns
Momentum 1M/3M  → Rate of price change (trend strength)
52W High/Low %  → Relative strength vs annual range
Vol Trend       → Volume expansion = institutional interest

Fundamental ratios feed the clustering engine directly.
──────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index via Wilder's smoothing.
    RSI > 70 → overbought | RSI < 30 → oversold
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD = EMA12 − EMA26   |   Signal = EMA9(MACD)   |   Histogram = MACD − Signal
    Histogram > 0 → bullish momentum
    """
    ema12    = compute_ema(series, 12)
    ema26    = compute_ema(series, 26)
    macd     = ema12 - ema26
    signal   = compute_ema(macd, 9)
    hist     = macd - signal
    return macd, signal, hist

def compute_bollinger_bands(series: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger %B = (Price − Lower) / (Upper − Lower)
    %B > 1 → above upper band (breakout candidate)
    %B < 0 → below lower band (mean-reversion candidate)
    """
    sma   = compute_sma(series, window)
    std   = series.rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct_b = (series - lower) / (upper - lower + 1e-10)
    return upper, lower, pct_b, std

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range — raw volatility in price units.
    Used for stop-loss sizing and risk normalisation.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_volume_trend(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume Z-score vs rolling mean — rising volume signals conviction."""
    avg = volume.rolling(window=window).mean()
    std = volume.rolling(window=window).std()
    return (volume - avg) / (std + 1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# FULL FEATURE PIPELINE  (per-ticker)
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  raw OHLCV DataFrame for a SINGLE ticker (date-indexed)
    Output: DataFrame with all technical features added
    """
    df = df.copy().sort_index()
    c  = df["close"]
    h  = df["high"]
    l  = df["low"]
    v  = df["volume"]

    # ── Trend Indicators ─────────────────────────────────────────────────────
    df["sma_20"]    = compute_sma(c, 20)
    df["sma_50"]    = compute_sma(c, 50)
    df["sma_200"]   = compute_sma(c, 200)
    df["ema_12"]    = compute_ema(c, 12)
    df["ema_26"]    = compute_ema(c, 26)

    # Price distance from moving averages (normalised %)
    df["price_vs_sma20"]  = (c - df["sma_20"])  / df["sma_20"]
    df["price_vs_sma50"]  = (c - df["sma_50"])  / df["sma_50"]
    df["price_vs_sma200"] = (c - df["sma_200"]) / df["sma_200"]

    # Golden/Death cross signal
    df["sma_50_200_cross"] = (df["sma_50"] - df["sma_200"]) / df["sma_200"]

    # ── Momentum Indicators ───────────────────────────────────────────────────
    df["rsi"]              = compute_rsi(c, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(c)
    df["macd_norm"]        = df["macd"] / c  # normalise by price

    df["momentum_1m"]  = c.pct_change(21)   # 21 trading days ≈ 1 month
    df["momentum_3m"]  = c.pct_change(63)   # 63 trading days ≈ 3 months
    df["momentum_6m"]  = c.pct_change(126)  # 126 trading days ≈ 6 months

    # ── Volatility ────────────────────────────────────────────────────────────
    log_ret            = np.log(c / c.shift(1))
    df["volatility_20d"] = log_ret.rolling(20).std() * np.sqrt(252)  # annualised
    df["volatility_60d"] = log_ret.rolling(60).std() * np.sqrt(252)

    df["atr"]            = compute_atr(h, l, c, 14)
    df["atr_pct"]        = df["atr"] / c   # ATR % of price

    bb_upper, bb_lower, df["bb_pct_b"], df["bb_width"] = compute_bollinger_bands(c, 20)
    df["bb_width_norm"]  = df["bb_width"] / c

    # ── 52-Week Range ─────────────────────────────────────────────────────────
    rolling_high         = c.rolling(252, min_periods=50).max()
    rolling_low          = c.rolling(252, min_periods=50).min()
    df["pct_from_52w_high"] = (c - rolling_high) / rolling_high  # ≤ 0
    df["pct_from_52w_low"]  = (c - rolling_low)  / rolling_low   # ≥ 0
    df["price_range_52w"]   = (rolling_high - rolling_low) / rolling_low

    # ── Volume ────────────────────────────────────────────────────────────────
    df["vol_zscore"]     = compute_volume_trend(v, 20)
    df["vol_sma_20"]     = compute_sma(v, 20)
    df["vol_ratio"]      = v / (df["vol_sma_20"] + 1)  # current vs avg volume

    # ── Daily Range (intraday volatility) ────────────────────────────────────
    df["daily_range"]    = (h - l) / c

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT FEATURES  (scalar per ticker for clustering)
# ══════════════════════════════════════════════════════════════════════════════

SNAPSHOT_TECH_FEATURES = [
    "rsi", "macd_norm", "macd_hist",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "sma_50_200_cross",
    "momentum_1m", "momentum_3m", "momentum_6m",
    "volatility_20d", "volatility_60d",
    "atr_pct", "bb_pct_b", "bb_width_norm",
    "pct_from_52w_high", "pct_from_52w_low",
    "vol_zscore", "vol_ratio",
]

FUNDAMENTAL_FEATURES = [
    "pe_ratio", "pb_ratio", "eps",
    "roe", "roa", "debt_to_equity",
    "revenue_growth", "profit_margin",
    "dividend_yield", "beta",
]

ALL_CLUSTER_FEATURES = SNAPSHOT_TECH_FEATURES + FUNDAMENTAL_FEATURES


def build_snapshot(df_prices_long: pd.DataFrame, df_fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker, collapse daily OHLCV → one row of features
    (latest technical snapshot + fundamentals).

    Parameters
    ----------
    df_prices_long   : long-format price dataframe with 'ticker' column
    df_fundamentals  : one row per ticker with fundamental metrics

    Returns
    -------
    snapshot_df : shape (n_tickers, n_features)
    """
    records = []

    for ticker, group in df_prices_long.groupby("ticker"):
        g = group.set_index("date").sort_index()
        g = engineer_features(g)

        # Take last valid row (most recent trading day)
        latest = g[SNAPSHOT_TECH_FEATURES].dropna().tail(1)
        if latest.empty:
            continue

        row = latest.iloc[0].to_dict()
        row["ticker"] = ticker
        records.append(row)

    snapshot_df = pd.DataFrame(records).set_index("ticker")

    # Merge fundamentals
    fund_sub = df_fundamentals.set_index("ticker")[FUNDAMENTAL_FEATURES]
    snapshot_df = snapshot_df.join(fund_sub, how="left")

    return snapshot_df


# ══════════════════════════════════════════════════════════════════════════════
# CLEANING + SCALING
# ══════════════════════════════════════════════════════════════════════════════

def clean_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    1. Winsorise outliers at 1st/99th percentile
    2. KNN-impute missing values (fundamental data has many NaNs)
    3. RobustScaler: median-centred, IQR-scaled → robust to outliers

    RobustScaler is preferred over StandardScaler here because
    financial data has heavy tails and frequent outliers.

    Returns
    -------
    df_clean  : cleaned (unscaled) dataframe
    df_scaled : scaled dataframe as DataFrame
    scaler    : fitted scaler object
    """
    # ── 1. Winsorise ─────────────────────────────────────────────────────────
    df_clean = df.copy()
    for col in df_clean.columns:
        lo = df_clean[col].quantile(0.01)
        hi = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(lo, hi)

    # ── 2. KNN Imputation ─────────────────────────────────────────────────────
    imputer  = KNNImputer(n_neighbors=5)
    arr_imp  = imputer.fit_transform(df_clean)
    df_clean = pd.DataFrame(arr_imp, index=df_clean.index, columns=df_clean.columns)

    # ── 3. Robust Scaling ─────────────────────────────────────────────────────
    scaler    = RobustScaler()
    arr_scaled = scaler.fit_transform(df_clean)
    df_scaled  = pd.DataFrame(arr_scaled, index=df_clean.index, columns=df_clean.columns)

    return df_clean, df_scaled, scaler


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_pipeline(
    prices_path:       str,
    fundamentals_path: str,
    output_path:       str,
) -> pd.DataFrame:
    """
    End-to-end feature pipeline.
    Reads CSVs → engineers features → cleans → saves processed snapshot.
    """
    df_prices = pd.read_csv(prices_path)
    df_fund   = pd.read_csv(fundamentals_path)

    df_prices["date"] = pd.to_datetime(df_prices["date"])

    snapshot          = build_snapshot(df_prices, df_fund)
    df_clean, df_scaled, scaler = clean_and_scale(snapshot[ALL_CLUSTER_FEATURES])

    # Attach metadata
    df_clean["sector"]   = df_fund.set_index("ticker").reindex(df_clean.index)["sector"]
    df_clean["industry"] = df_fund.set_index("ticker").reindex(df_clean.index)["industry"]
    df_clean["name"]     = df_fund.set_index("ticker").reindex(df_clean.index)["name"]

    df_clean.to_csv(output_path)
    print(f"[FeaturePipeline] Saved {df_clean.shape} → {output_path}")

    return df_clean, df_scaled, scaler


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-TICKER FEATURES  (Streamlit app)
# ══════════════════════════════════════════════════════════════════════════════

def get_ticker_features(ohlcv_df: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
    """
    Given a single ticker's OHLCV df + fundamentals dict,
    return a 1-row feature snapshot (unscaled).
    """
    ohlcv_df = ohlcv_df.set_index("date") if "date" in ohlcv_df.columns else ohlcv_df
    featured = engineer_features(ohlcv_df)
    latest   = featured[SNAPSHOT_TECH_FEATURES].dropna().tail(1)

    if latest.empty:
        return pd.DataFrame()

    row = latest.iloc[0].to_dict()
    for k in FUNDAMENTAL_FEATURES:
        row[k] = fundamentals.get(k, np.nan)

    return pd.DataFrame([row])


if __name__ == "__main__":
    import os
    base = os.path.join(os.path.dirname(__file__), "..")

    # STEP 3 — Feature pipeline
    df_clean, df_scaled, scaler = run_feature_pipeline(
        prices_path       = os.path.join(base, "data/raw/SP500_prices.csv"),
        fundamentals_path = os.path.join(base, "data/processed/SP500_fundamentals.csv"),
        output_path       = os.path.join(base, "data/processed/stocks.csv"),
    )

    print("\nFeature matrix shape:", df_clean.shape)
    print(df_clean.head())

    # ═════════════════════════════════════════════════════
    # STEP 4 — KMEANS CLUSTERING  ⭐ ADD HERE
    # ═════════════════════════════════════════════════════
    from sklearn.cluster import KMeans

    print("\n[Clustering] Running KMeans...")

    X_cluster = df_scaled.copy()

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    kmeans.fit(X_cluster)

    df_clean["cluster"] = kmeans.labels_

    print("\nCluster distribution:")
    print(df_clean["cluster"].value_counts())

    cluster_output_path = os.path.join(base, "data/processed/stocks_clustered.csv")
    df_clean.to_csv(cluster_output_path)

    print(f"\n[Clustering] Saved clustered dataset → {cluster_output_path}")


