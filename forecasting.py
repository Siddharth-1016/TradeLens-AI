"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 7 — PRICE FORECASTING MODULE                          ║
║  Prophet (default) + LSTM fallback → 30-day price forecast  ║
╚══════════════════════════════════════════════════════════════╝

MODEL CHOICE RATIONALE:
────────────────────────────────────────────────────────────────
Prophet (Facebook/Meta):
  ✅ Handles seasonality (weekly, yearly trading patterns)
  ✅ Robust to missing data and outliers (common in stock data)
  ✅ Uncertainty intervals (confidence bounds) built-in
  ✅ No normalisation needed
  ✅ Fast training (seconds per ticker)
  ⚠️  Not recurrent → doesn't capture path-dependency

LSTM (Fallback):
  ✅ Captures non-linear sequential dependencies
  ✅ Better for regime-change periods
  ⚠️  Needs 200+ data points, slower training, hyperparameter-sensitive

We default to Prophet and fall back to a simple Linear Regression
if Prophet is not installed (for leaner deployments).
────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[Forecasting] Prophet not installed — using LinearRegression fallback")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler


# ══════════════════════════════════════════════════════════════════════════════
# PROPHET FORECASTER
# ══════════════════════════════════════════════════════════════════════════════

def forecast_prophet(
    df_prices:  pd.DataFrame,
    periods:    int = 30,
    ticker:     str = "",
) -> dict:
    """
    Train Prophet on historical close prices and forecast `periods` days ahead.

    Returns
    -------
    dict with:
      forecast_df   : full Prophet forecast DataFrame
      future_df     : just the forecasted period (next 30 days)
      direction     : "Up" / "Down" / "Sideways"
      confidence    : float 0-1
      pct_change    : expected % price change
    """
    if not PROPHET_AVAILABLE:
        return forecast_fallback(df_prices, periods, ticker)

    # Prophet requires columns: ds (date), y (value)
    price_col = "close" if "close" in df_prices.columns else "Close"
    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(df_prices.index if df_prices.index.name == "date"
                             else df_prices.get("date", df_prices.index)),
        "y":  df_prices[price_col].values,
    }).dropna()

    # ── Model config ─────────────────────────────────────────────────────────
    model = Prophet(
        changepoint_prior_scale   = 0.05,     # flexibility of trend changes
        seasonality_prior_scale   = 10.0,     # flexibility of seasonality
        weekly_seasonality        = True,
        yearly_seasonality        = True,
        daily_seasonality         = False,
        interval_width            = 0.80,     # 80% confidence interval
        uncertainty_samples       = 500,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    model.fit(prophet_df, )

    # ── Future dataframe ─────────────────────────────────────────────────────
    future       = model.make_future_dataframe(periods=periods, freq="B")  # business days
    forecast     = model.predict(future)

    future_only  = forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    last_actual  = prophet_df["y"].iloc[-1]
    last_forecast = future_only["yhat"].iloc[-1]
    pct_change   = (last_forecast - last_actual) / last_actual

    # ── Direction & Confidence ────────────────────────────────────────────────
    if pct_change > 0.02:
        direction = "📈 Up"
    elif pct_change < -0.02:
        direction = "📉 Down"
    else:
        direction = "➡️ Sideways"

    # Confidence: inversely proportional to the width of the CI
    ci_width  = (future_only["yhat_upper"] - future_only["yhat_lower"]).mean()
    rel_width = ci_width / abs(last_forecast) if last_forecast != 0 else 1.0
    confidence = max(0.0, min(1.0, 1 - rel_width))

    print(f"[Prophet] {ticker} → {direction}  ({pct_change:+.2%})  "
          f"Confidence={confidence:.2%}")

    return {
        "forecast_df":  forecast,
        "future_df":    future_only,
        "direction":    direction,
        "confidence":   confidence,
        "pct_change":   pct_change,
        "last_actual":  last_actual,
        "last_forecast": last_forecast,
        "model":        "Prophet",
    }


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK FORECASTER  (Ridge Regression on lagged features)
# ══════════════════════════════════════════════════════════════════════════════

def forecast_fallback(
    df_prices: pd.DataFrame,
    periods:   int = 30,
    ticker:    str = "",
) -> dict:
    """
    Lightweight fallback when Prophet is not available.
    Uses Ridge Regression on lag features + rolling mean.
    """
    price_col = "close" if "close" in df_prices.columns else "Close"
    series    = df_prices[price_col].dropna().values.reshape(-1, 1)

    scaler = MinMaxScaler()
    norm   = scaler.fit_transform(series).flatten()

    LAG = 30
    X, y = [], []
    for i in range(LAG, len(norm)):
        X.append(norm[i-LAG:i])
        y.append(norm[i])
    X, y = np.array(X), np.array(y)

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Walk-forward forecast
    window      = list(norm[-LAG:])
    predictions = []
    for _ in range(periods):
        x_input = np.array(window[-LAG:]).reshape(1, -1)
        pred    = model.predict(x_input)[0]
        predictions.append(pred)
        window.append(pred)

    pred_prices = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    last_actual   = series[-1, 0]
    last_forecast = pred_prices[-1]
    pct_change    = (last_forecast - last_actual) / last_actual

    if pct_change > 0.02:
        direction = "📈 Up"
    elif pct_change < -0.02:
        direction = "📉 Down"
    else:
        direction = "➡️ Sideways"

    confidence = 0.55  # lower confidence for fallback model

    # Build future_df
    last_date  = pd.to_datetime(
        df_prices.index[-1] if df_prices.index.name == "date"
        else df_prices.index[-1]
    )
    date_range = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    future_df  = pd.DataFrame({
        "ds":          date_range,
        "yhat":        pred_prices,
        "yhat_lower":  pred_prices * 0.97,
        "yhat_upper":  pred_prices * 1.03,
    })

    print(f"[Ridge Fallback] {ticker} → {direction}  ({pct_change:+.2%})")

    return {
        "forecast_df":   future_df,
        "future_df":     future_df,
        "direction":     direction,
        "confidence":    confidence,
        "pct_change":    pct_change,
        "last_actual":   last_actual,
        "last_forecast": last_forecast,
        "model":         "Ridge (fallback)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE LSTM  (optional, heavier — uses TensorFlow/Keras)
# ══════════════════════════════════════════════════════════════════════════════

def forecast_lstm(
    df_prices: pd.DataFrame,
    periods:   int = 30,
    ticker:    str = "",
    lookback:  int = 60,
) -> dict:
    """
    LSTM-based forecaster. Requires tensorflow/keras.

    Architecture: LSTM(50) → Dropout(0.2) → LSTM(50) → Dense(1)
    Trained for 30 epochs with EarlyStopping.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("[LSTM] TensorFlow not available — using fallback")
        return forecast_fallback(df_prices, periods, ticker)

    price_col = "close" if "close" in df_prices.columns else "Close"
    raw       = df_prices[price_col].dropna().values.reshape(-1, 1)

    if len(raw) < lookback + 50:
        return forecast_fallback(df_prices, periods, ticker)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw)

    # Build sequences
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)

    # ── Model ────────────────────────────────────────────────────────────────
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=16, validation_split=0.1,
              callbacks=[es], verbose=0)

    # Walk-forward prediction
    seq       = list(scaled[-lookback:, 0])
    preds     = []
    for _ in range(periods):
        inp  = np.array(seq[-lookback:]).reshape(1, lookback, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        preds.append(pred)
        seq.append(pred)

    pred_prices = scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()

    last_actual   = raw[-1, 0]
    last_forecast = pred_prices[-1]
    pct_change    = (last_forecast - last_actual) / last_actual

    direction = (
        "📈 Up"      if pct_change > 0.02 else
        "📉 Down"    if pct_change < -0.02 else
        "➡️ Sideways"
    )
    confidence = 0.65  # LSTM uncertainty estimate

    last_date  = pd.to_datetime(df_prices.index[-1])
    date_range = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    future_df  = pd.DataFrame({
        "ds":         date_range,
        "yhat":       pred_prices,
        "yhat_lower": pred_prices * 0.96,
        "yhat_upper": pred_prices * 1.04,
    })

    print(f"[LSTM] {ticker} → {direction}  ({pct_change:+.2%})")
    return {
        "forecast_df":   future_df,
        "future_df":     future_df,
        "direction":     direction,
        "confidence":    confidence,
        "pct_change":    pct_change,
        "last_actual":   last_actual,
        "last_forecast": last_forecast,
        "model":         "LSTM",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MASTER FORECAST FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_forecast(
    df_prices:  pd.DataFrame,
    ticker:     str = "",
    method:     str = "prophet",  # "prophet" | "lstm" | "fallback"
    periods:    int = 30,
) -> dict:
    """
    Dispatch to the appropriate forecasting method.
    """
    df_prices = df_prices.copy()
    if "date" in df_prices.columns:
        df_prices = df_prices.set_index("date")
    df_prices.index = pd.to_datetime(df_prices.index)
    df_prices        = df_prices.sort_index()

    if method == "prophet" and PROPHET_AVAILABLE:
        return forecast_prophet(df_prices, periods, ticker)
    elif method == "lstm":
        return forecast_lstm(df_prices, periods, ticker)
    else:
        return forecast_fallback(df_prices, periods, ticker)


if __name__ == "__main__":
    from data_collection import fetch_ohlcv
    df = fetch_ohlcv("AAPL")
    if df is not None:
        result = run_forecast(df, "AAPL")
        print(f"\nForecast: {result['direction']} | {result['pct_change']:+.2%} | "
              f"Confidence: {result['confidence']:.2%}")
        print(result["future_df"].tail(5))
