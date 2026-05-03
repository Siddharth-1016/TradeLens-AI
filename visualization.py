"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 10 — VISUALIZATION DASHBOARD (Plotly)                 ║
║  PCA scatter, radar, forecast chart, sentiment gauge        ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Theme constants ─────────────────────────────────────────────────────────
DARK_BG    = "#0D1117"
CARD_BG    = "#161B22"
GRID_COLOR = "#30363D"
TEXT_COLOR = "#E6EDF3"
ACCENT_1   = "#58A6FF"  # Blue
ACCENT_2   = "#3FB950"  # Green
ACCENT_3   = "#F85149"  # Red
ACCENT_4   = "#E3B341"  # Yellow

LABEL_COLORS = {
    "🟢 BUY":      "#3FB950",
    "🟡 MAYBE BUY": "#E3B341",
    "🔴 NOT BUY":  "#F85149",
}

DECISION_COLORS = {
    "🚀 STRONG BUY": "#00E676",
    "🟢 BUY":        "#3FB950",
    "🟡 HOLD":       "#E3B341",
    "🔴 AVOID":      "#F85149",
}

BASE_LAYOUT = dict(
    paper_bgcolor = DARK_BG,
    plot_bgcolor  = DARK_BG,
    font          = dict(family="JetBrains Mono, monospace", color=TEXT_COLOR, size=12),
    xaxis         = dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
    yaxis         = dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
    margin        = dict(t=50, b=40, l=40, r=20),
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. PCA CLUSTER SCATTER
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca_clusters(
    X_pca:       np.ndarray,
    df_labelled: pd.DataFrame,
    pca_obj,
    title:       str = "Stock Clusters — PCA Projection",
) -> go.Figure:
    """
    2D scatter plot of stocks coloured by investment label.
    Hover shows ticker, cluster, and key metrics.
    """
    df_plot = df_labelled.copy().reset_index()
    df_plot["pc1"]   = X_pca[:, 0]
    df_plot["pc2"]   = X_pca[:, 1]

    ev1 = pca_obj.explained_variance_ratio_[0] * 100
    ev2 = pca_obj.explained_variance_ratio_[1] * 100

    fig = go.Figure()

    for label, color in LABEL_COLORS.items():
        mask = df_plot["investment_label"] == label
        sub  = df_plot[mask]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x    = sub["pc1"],
            y    = sub["pc2"],
            mode = "markers+text",
            name = label,
            text = sub.get("ticker", sub.index).values,
            textposition = "top center",
            textfont     = dict(size=8, color=color),
            marker       = dict(
                color   = color,
                size    = 10,
                opacity = 0.85,
                line    = dict(width=1, color=DARK_BG),
                symbol  = "circle",
            ),
            hovertemplate = (
                "<b>%{text}</b><br>"
                f"Label: {label}<br>"
                "PC1: %{x:.3f} | PC2: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title  = dict(text=title, font=dict(size=15, color=ACCENT_1)),
        xaxis_title = f"PC1 ({ev1:.1f}% var.)",
        yaxis_title = f"PC2 ({ev2:.1f}% var.)",
        legend = dict(bgcolor=CARD_BG, bordercolor=GRID_COLOR, borderwidth=1),
        **BASE_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. RADAR CHART — CLUSTER AVERAGES
# ══════════════════════════════════════════════════════════════════════════════

RADAR_FEATURES = [
    "roe", "profit_margin", "momentum_3m",
    "rsi_norm", "volatility_inv", "revenue_growth",
]

RADAR_LABELS = [
    "ROE", "Profit Margin", "3M Momentum",
    "RSI Score", "Low Volatility", "Revenue Growth",
]


def _normalise_radar(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise each feature to [0, 1] for radar display."""
    df = df.copy()
    # RSI: closer to 50 = score 1, extremes = 0
    if "rsi" in df.columns:
        df["rsi_norm"] = 1 - abs(df["rsi"] - 50) / 50
    # Volatility: invert (low vol = high score)
    if "volatility_20d" in df.columns:
        mn, mx = df["volatility_20d"].min(), df["volatility_20d"].max()
        df["volatility_inv"] = 1 - (df["volatility_20d"] - mn) / (mx - mn + 1e-8)

    for col in ["roe", "profit_margin", "momentum_3m", "revenue_growth"]:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            df[col] = (df[col] - mn) / (mx - mn + 1e-8)

    return df


def plot_radar_clusters(df_labelled: pd.DataFrame) -> go.Figure:
    """
    Radar chart comparing cluster averages on 6 key dimensions.
    """
    df_norm    = _normalise_radar(df_labelled)
    available  = [f for f in RADAR_FEATURES if f in df_norm.columns]
    feat_labels = [RADAR_LABELS[RADAR_FEATURES.index(f)] for f in available]

    fig = go.Figure()

    for label, color in LABEL_COLORS.items():
        mask  = df_norm["investment_label"] == label
        sub   = df_norm[mask]
        if sub.empty:
            continue
        values = [sub[f].mean() for f in available]
        values += [values[0]]  # close the polygon

        fig.add_trace(go.Scatterpolar(
            r     = values,
            theta = feat_labels + [feat_labels[0]],
            name  = label,
            fill  = "toself",
            line  = dict(color=color, width=2),
            fillcolor = color,
            opacity = 0.3,
        ))

    fig.update_layout(
        polar  = dict(
            bgcolor    = CARD_BG,
            radialaxis = dict(visible=True, range=[0, 1],
                              gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
            angularaxis= dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
        ),
        title  = dict(text="Cluster Profile Radar", font=dict(size=15, color=ACCENT_1)),
        legend = dict(bgcolor=CARD_BG, bordercolor=GRID_COLOR),
        **{k: v for k, v in BASE_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. PRICE FORECAST CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecast(
    df_prices:      pd.DataFrame,
    forecast_result: dict,
    ticker:         str,
) -> go.Figure:
    """
    Combined historical + forecast price chart with confidence bands.
    """
    price_col = "close" if "close" in df_prices.columns else "Close"
    df_hist   = df_prices.copy()

    if "date" in df_hist.columns:
        df_hist = df_hist.set_index("date")
    df_hist.index = pd.to_datetime(df_hist.index)
    df_hist = df_hist.sort_index().tail(180)  # last 6 months for context

    future_df = forecast_result.get("future_df", pd.DataFrame())
    direction = forecast_result.get("direction", "")
    pct_chg   = forecast_result.get("pct_change", 0)
    model_name= forecast_result.get("model", "Forecast")

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x    = df_hist.index,
        y    = df_hist[price_col],
        name = "Historical Price",
        line = dict(color=ACCENT_1, width=2),
        hovertemplate = "%{x|%b %d %Y}: $%{y:.2f}<extra></extra>",
    ))

    # Forecast
    if not future_df.empty:
        ds_col = "ds" if "ds" in future_df.columns else future_df.columns[0]
        y_col  = "yhat"

        # Confidence band
        if "yhat_upper" in future_df.columns:
            fig.add_trace(go.Scatter(
                x    = pd.concat([future_df[ds_col], future_df[ds_col].iloc[::-1]]),
                y    = pd.concat([future_df["yhat_upper"], future_df["yhat_lower"].iloc[::-1]]),
                fill = "toself",
                fillcolor = "rgba(88,166,255,0.12)",
                line = dict(width=0),
                name = "80% CI",
                showlegend = True,
                hoverinfo  = "skip",
            ))

        # Forecast line
        dir_color = ACCENT_2 if "Up" in direction else (ACCENT_3 if "Down" in direction else ACCENT_4)
        fig.add_trace(go.Scatter(
            x    = future_df[ds_col],
            y    = future_df[y_col],
            name = f"{model_name} ({pct_chg:+.1%})",
            line = dict(color=dir_color, width=2.5, dash="dot"),
            hovertemplate = "%{x|%b %d %Y}: $%{y:.2f}<extra></extra>",
        ))

        # Connect historical to forecast
        last_hist_date  = df_hist.index[-1]
        last_hist_price = df_hist[price_col].iloc[-1]
        first_fore_date = future_df[ds_col].iloc[0]
        first_fore_price= future_df[y_col].iloc[0]
        fig.add_trace(go.Scatter(
            x    = [last_hist_date, first_fore_date],
            y    = [last_hist_price, first_fore_price],
            line = dict(color=dir_color, width=1.5, dash="dot"),
            showlegend = False,
            hoverinfo  = "skip",
        ))

    fig.update_layout(
        title  = dict(text=f"{ticker} — 30-Day Price Forecast  |  {direction}",
                      font=dict(size=15, color=ACCENT_1)),
        xaxis_title = "Date",
        yaxis_title = "Price (USD)",
        hovermode   = "x unified",
        legend      = dict(bgcolor=CARD_BG, bordercolor=GRID_COLOR),
        **BASE_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. SENTIMENT GAUGE
# ══════════════════════════════════════════════════════════════════════════════

def plot_sentiment_gauge(
    gauge_value:  int,
    label:        str,
    ticker:       str,
) -> go.Figure:
    """
    Gauge chart from 0 (very negative) → 100 (very positive).
    """
    clean_label = label.split(" ", 1)[-1]
    color = (ACCENT_2 if "Positive" in label
             else ACCENT_3 if "Negative" in label
             else ACCENT_4)

    fig = go.Figure(go.Indicator(
        mode     = "gauge+number+delta",
        value    = gauge_value,
        title    = {"text": f"{ticker} — News Sentiment", "font": {"size": 14, "color": TEXT_COLOR}},
        delta    = {"reference": 50, "suffix": " vs neutral",
                    "increasing": {"color": ACCENT_2}, "decreasing": {"color": ACCENT_3}},
        gauge    = {
            "axis":      {"range": [0, 100], "tickwidth": 1, "tickcolor": TEXT_COLOR},
            "bar":       {"color": color, "thickness": 0.25},
            "bgcolor":   CARD_BG,
            "borderwidth": 1,
            "bordercolor": GRID_COLOR,
            "steps": [
                {"range": [0,  33], "color": "#1A0A0A"},
                {"range": [33, 66], "color": "#1A1A0A"},
                {"range": [66, 100],"color": "#0A1A0A"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 4},
                "thickness": 0.8,
                "value": gauge_value,
            },
        },
        number   = {"suffix": " / 100", "font": {"color": color, "size": 28}},
    ))

    fig.update_layout(
        height  = 280,
        **{k: v for k, v in BASE_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. SHAP BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_importance(shap_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of feature importance (mean |SHAP|).
    """
    df = shap_df.copy().sort_values("shap_value")

    colors = [ACCENT_2 if v >= 0 else ACCENT_3 for v in df["shap_value"]]

    fig = go.Figure(go.Bar(
        x           = df["shap_value"],
        y           = df["feature"].apply(lambda f: f.replace("_", " ").title()),
        orientation = "h",
        marker_color= colors,
        hovertemplate = "<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title  = dict(text="SHAP Feature Contributions", font=dict(size=15, color=ACCENT_1)),
        xaxis_title = "SHAP Value  (positive → BUY | negative → NOT BUY)",
        height = max(350, len(df) * 22),
        **BASE_LAYOUT,
    )
    fig.add_vline(x=0, line_width=1, line_color=GRID_COLOR)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. DECISION SCORE BREAKDOWN BAR
# ══════════════════════════════════════════════════════════════════════════════

def plot_decision_breakdown(decision) -> go.Figure:
    """
    Stacked bar showing contribution of each component to final score.
    """
    components = ["Cluster Signal", "Forecast Signal", "Sentiment Signal"]
    values     = [
        decision.cluster_score,
        decision.forecast_score,
        decision.sentiment_w_score,
    ]
    colors = [ACCENT_2 if v >= 0 else ACCENT_3 for v in values]

    fig = go.Figure(go.Bar(
        x           = components,
        y           = values,
        marker_color= colors,
        text        = [f"{v:+.3f}" for v in values],
        textposition= "outside",
        hovertemplate = "<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
    ))

    total = decision.total_score
    rec   = decision.recommendation
    fig.add_hline(y=0, line_width=1, line_color=GRID_COLOR)
    fig.update_layout(
        title = dict(
            text=f"Score Breakdown — {rec}  (Total: {total:+.4f})",
            font=dict(size=15, color=DECISION_COLORS.get(rec, TEXT_COLOR)),
        ),
        yaxis_title = "Weighted Score",
        yaxis_range = [-0.6, 0.7],
        **BASE_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. LEADERBOARD TABLE
# ══════════════════════════════════════════════════════════════════════════════

def plot_leaderboard(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Styled table of top stock recommendations."""
    df = df.head(top_n).copy()

    # Color-code recommendation column
    cell_colors = [
        [DECISION_COLORS.get(r, TEXT_COLOR) for r in df["recommendation"]]
        if col == "recommendation" else [TEXT_COLOR] * len(df)
        for col in df.columns
    ]

    fig = go.Figure(go.Table(
        header = dict(
            values    = [f"<b>{c.upper()}</b>" for c in df.columns],
            fill_color= ACCENT_1,
            align     = "left",
            font      = dict(color="black", size=11, family="monospace"),
            height    = 30,
        ),
        cells = dict(
            values    = [df[c].tolist() for c in df.columns],
            fill_color= CARD_BG,
            align     = "left",
            font      = dict(color=cell_colors, size=10, family="monospace"),
            height    = 26,
        ),
    ))

    fig.update_layout(
        title  = dict(text="📊 Investment Leaderboard", font=dict(size=15, color=ACCENT_1)),
        margin = dict(t=50, b=20, l=10, r=10),
        **{k: v for k, v in BASE_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8. CANDLESTICK + VOLUME
# ══════════════════════════════════════════════════════════════════════════════

def plot_candlestick(df_prices: pd.DataFrame, ticker: str, days: int = 90) -> go.Figure:
    """Candlestick chart with volume bars for recent price history."""
    df = df_prices.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().tail(days)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x     = df.index,
        open  = df["open"],  high  = df["high"],
        low   = df["low"],   close = df["close"],
        name  = ticker,
        increasing_line_color = ACCENT_2,
        decreasing_line_color = ACCENT_3,
    ), row=1, col=1)

    # Volume
    colors = [ACCENT_2 if c >= o else ACCENT_3
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x             = df.index,
        y             = df["volume"],
        name          = "Volume",
        marker_color  = colors,
        opacity       = 0.7,
    ), row=2, col=1)

    fig.update_layout(
        title       = dict(text=f"{ticker} — {days}d Price Chart", font=dict(size=15, color=ACCENT_1)),
        xaxis_rangeslider_visible = False,
        legend      = dict(bgcolor=CARD_BG),
        **BASE_LAYOUT,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor=GRID_COLOR)
    return fig
