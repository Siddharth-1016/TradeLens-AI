"""
STEP 6 — EXPLAINABLE AI MODULE (SHAP) — FULLY FIXED (XGBoost v2 Compatible)
"""

import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")

FEATURE_PHRASES = {
    "roe": ("high ROE (efficient capital use)", "low ROE"),
    "roa": ("strong ROA (asset efficiency)", "weak ROA"),
    "debt_to_equity": ("low debt/equity ratio", "high debt burden"),
    "profit_margin": ("healthy profit margins", "weak profit margins"),
    "revenue_growth": ("strong revenue growth", "declining revenue"),
    "pe_ratio": ("reasonable P/E valuation", "expensive P/E valuation"),
    "pb_ratio": ("low price-to-book (value play)", "high price-to-book"),
    "eps": ("positive EPS", "negative / weak EPS"),
    "dividend_yield": ("attractive dividend yield", "low dividend yield"),
    "beta": ("low beta (stable)", "high beta (volatile)"),
    "rsi": ("healthy RSI momentum", "overbought/oversold RSI"),
    "macd_hist": ("positive MACD histogram", "negative MACD histogram"),
    "momentum_1m": ("positive 1-month momentum", "negative 1-month momentum"),
    "momentum_3m": ("strong 3-month trend", "weak 3-month trend"),
    "momentum_6m": ("strong 6-month trend", "weak 6-month trend"),
    "price_vs_sma50": ("price above 50-day MA", "price below 50-day MA"),
    "price_vs_sma200": ("price above 200-day MA", "price below 200-day MA"),
    "sma_50_200_cross": ("golden cross (bullish)", "death cross (bearish)"),
    "volatility_20d": ("low short-term volatility", "high short-term volatility"),
    "pct_from_52w_high": ("near 52-week high", "far below 52-week high"),
    "bb_pct_b": ("positive Bollinger position", "below Bollinger lower band"),
    "vol_zscore": ("above-average volume", "below-average volume"),
}

from xgboost import XGBClassifier


# ═══════════════════════════════════════════════════════════════
# TRAIN PROXY MODEL
# ═══════════════════════════════════════════════════════════════

def train_proxy_model(df_scaled, labels, features):
    """
    Train XGBoost proxy classifier and create SHAP explainer
    Compatible with XGBoost v2 + NumPy 2 + SHAP latest
    """

    X = df_scaled[features].fillna(0).values

    # Convert clusters → binary BUY vs NOT BUY
    # (cluster 2 assumed best cluster)
    y = (labels == 2).astype(int)

    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    )

    model.fit(X, y)

    acc = (model.predict(X) == y).mean()
    print(f"[SHAP] Proxy model accuracy: {acc:.4f}")

    # ⭐⭐ THE MAGIC LINE ⭐⭐
    # Use model.predict_proba instead of TreeExplainer
    explainer = shap.Explainer(
        model.predict_proba,
        X,                  # background dataset
        feature_names=features
    )

    shap_values = explainer(X)

    return model, explainer, shap_values


# ═══════════════════════════════════════════════════════════════
# SINGLE STOCK EXPLANATION
# ═══════════════════════════════════════════════════════════════

def explain_single(ticker_features, explainer, scaler, features, cluster_id, label_map):
    """
    Generate SHAP explanation for ONE stock.
    Works with binary XGBoost proxy model.
    """

    # Prepare data
    X_raw = ticker_features[features].fillna(0).values
    X_scaled = scaler.transform(X_raw)

    # SHAP values (binary model → single array returned)
    shap_vals = explainer.shap_values(X_scaled)

    # shap_vals shape = (1, n_features)
    class_shap = shap_vals[0,:,1]

    # Create dataframe
    shap_df = pd.DataFrame({
        "feature": features,
        "shap_value": class_shap,
        "raw_value": X_raw[0],
    })

    # Sort by importance
    shap_df = shap_df.reindex(
        shap_df["shap_value"].abs().sort_values(ascending=False).index
    )

    # Direction column
    shap_df["direction"] = shap_df["shap_value"].apply(
        lambda v: "↑ Positive (pushes toward BUY)" if v > 0 
        else "↓ Negative (pushes away from BUY)"
    )

    return shap_df


# ═══════════════════════════════════════════════════════════════
# GLOBAL FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def get_shap_summary_data(shap_values, features):
    """
    Returns mean absolute SHAP importance per feature.
    Compatible with new SHAP Explanation object.
    """

    # Extract numpy array from Explanation object
    values = shap_values.values

    # values shape = (samples, features, classes)
    # we care about BUY class → index 1
    buy_class_values = values[:, :, 1]

    # mean absolute SHAP per feature
    mean_abs = np.abs(buy_class_values).mean(axis=0)

    df = pd.DataFrame({
        "feature": features,
        "importance": mean_abs
    }).sort_values("importance", ascending=True).tail(15)

    # Human readable names (safe fallback)
    df["feature_label"] = df["feature"].apply(
        lambda f: FEATURE_PHRASES.get(f, (f,))[0]
    )

    return df


# ═══════════════════════════════════════════════════════════════
# TEST RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_engineering import ALL_CLUSTER_FEATURES, clean_and_scale

    base = os.path.join(os.path.dirname(__file__), "..")
    df   = pd.read_csv(os.path.join(base, "data/processed/stocks.csv"), index_col="ticker")

    feats = [f for f in ALL_CLUSTER_FEATURES if f in df.columns]
    _, df_scaled, scaler = clean_and_scale(df[feats])

    labels = np.random.randint(0, 3, len(df))

    model, explainer, shap_values = train_proxy_model(df_scaled, labels, feats)

    print("✅ SHAP model trained successfully!")
    print("SHAP shape:", np.array(shap_values).shape)

    summary = get_shap_summary_data(shap_values, feats)
    print(summary.head())