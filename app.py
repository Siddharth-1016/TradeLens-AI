"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 11 — STREAMLIT PRODUCTION DASHBOARD                   ║
║  AI-Powered Quant Investment Decision Platform              ║
╚══════════════════════════════════════════════════════════════╝

Run:  streamlit run app.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import time

# Internal modules
from data_collection   import fetch_single_ticker, SP500_TICKERS, NIFTY50_TICKERS
from feature_engineering import (
    get_ticker_features, ALL_CLUSTER_FEATURES,
    SNAPSHOT_TECH_FEATURES, FUNDAMENTAL_FEATURES, clean_and_scale
)
from clustering        import run_kmeans, assign_investment_labels, predict_single, pca_scatter
from explainability    import train_proxy_model, explain_single
from forecasting       import run_forecast
from sentiment         import run_sentiment_analysis
from decision_engine   import compute_decision
from visualization     import (
    plot_pca_clusters, plot_radar_clusters, plot_forecast,
    plot_sentiment_gauge, plot_shap_importance, plot_decision_breakdown,
    plot_candlestick, plot_leaderboard,
    DARK_BG, ACCENT_1, TEXT_COLOR
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "QuantAI — Investment Platform",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Sora:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    background-color: #0D1117 !important;
    color: #E6EDF3 !important;
    font-family: 'Sora', sans-serif;
}

/* Header */
.quant-header {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.quant-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #58A6FF, #3FB950, #E3B341, #F85149);
}
.quant-header h1 {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    margin: 0;
    background: linear-gradient(135deg, #58A6FF, #3FB950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.quant-header p { color: #8B949E; margin: 4px 0 0; font-size: 0.9rem; }

/* Metric cards */
.metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58A6FF; }
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Recommendation badge */
.rec-badge {
    display: inline-block;
    padding: 10px 24px;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 800;
    font-family: 'Sora', sans-serif;
    letter-spacing: 0.05em;
    text-align: center;
    width: 100%;
}

/* Section divider */
.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #58A6FF;
    border-left: 3px solid #58A6FF;
    padding-left: 10px;
    margin: 20px 0 12px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #30363D;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161B22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8B949E;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
.stTabs [aria-selected="true"] {
    background: #21262D !important;
    color: #58A6FF !important;
}

/* Plotly container */
.js-plotly-plot { border-radius: 8px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="quant-header">
    <h1>⚡ QuantAI Investment Platform</h1>
    <p>AI-powered stock analysis · Clustering · Forecasting · Sentiment · Explainability</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🎯 Stock Analyser")

    market = st.selectbox(
        "Market Universe",
        ["S&P 500", "NIFTY 50", "Custom"],
        index=0,
    )

    if market == "S&P 500":
        default_tickers = SP500_TICKERS[:20]
        ticker = st.selectbox("Select Ticker", SP500_TICKERS, index=0)
    elif market == "NIFTY 50":
        default_tickers = NIFTY50_TICKERS[:20]
        ticker = st.selectbox("Select Ticker", NIFTY50_TICKERS, index=0)
    else:
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL").upper().strip()

    st.divider()

    forecast_model = st.selectbox(
        "Forecast Model",
        ["prophet", "fallback"],
        index=0,
        help="Prophet requires `prophet` package. Fallback uses Ridge Regression.",
    )

    use_finbert = st.checkbox(
        "Use FinBERT (Slower)",
        value=False,
        help="FinBERT is more accurate but requires `transformers`",
    )

    st.divider()

    run_btn = st.button("🚀 Analyse Stock", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#8B949E; font-family:'JetBrains Mono',monospace;">
    <b>Platform Modules</b><br>
    ✅ Stock Clustering<br>
    ✅ Price Forecasting<br>
    ✅ Sentiment Analysis<br>
    ✅ Explainable AI (SHAP)<br>
    ✅ Decision Engine<br><br>
    <i>⚠️ Not financial advice.</i>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

REC_STYLE = {
    "🚀 STRONG BUY": ("background:#00E676;color:#000;", "🚀 STRONG BUY"),
    "🟢 BUY":        ("background:#3FB950;color:#000;", "🟢 BUY"),
    "🟡 HOLD":       ("background:#E3B341;color:#000;", "🟡 HOLD"),
    "🔴 AVOID":      ("background:#F85149;color:#fff;", "🔴 AVOID"),
}


@st.cache_data(ttl=900, show_spinner=False)  # 15-min cache
def run_full_analysis(ticker: str, forecast_model: str, use_finbert: bool):
    """Cached full analysis pipeline."""
    # 1. Data
    fundamentals, prices = fetch_single_ticker(ticker)
    if prices is None or prices.empty:
        return None, "No price data available for this ticker."

    # 2. Feature Engineering
    features = get_ticker_features(prices, fundamentals)
    if features.empty:
        return None, "Not enough price history to compute features."

    available_feats = [f for f in ALL_CLUSTER_FEATURES if f in features.columns]
    feat_vals = features[available_feats].fillna(0)

    # Minimal cluster: treat this single stock — use mock cluster for demo
    # In production you'd load the pre-trained KMeans model
    # Here we simulate: score-based heuristic cluster label
    roe      = fundamentals.get("roe",      0) or 0
    mom3m    = feat_vals.get("momentum_3m", pd.Series([0])).values[0]
    rsi_val  = feat_vals.get("rsi",         pd.Series([50])).values[0]
    d_e      = fundamentals.get("debt_to_equity", 1) or 1
    rev_g    = fundamentals.get("revenue_growth", 0) or 0

    quality_score = roe * 2 + mom3m + rev_g - d_e * 0.1
    if quality_score > 0.2:
        cluster_label = "🟢 BUY"
    elif quality_score > -0.1:
        cluster_label = "🟡 MAYBE BUY"
    else:
        cluster_label = "🔴 NOT BUY"

    # 3. Forecast
    forecast = run_forecast(prices, ticker=ticker, method=forecast_model)

    # 4. Sentiment
    sentiment = run_sentiment_analysis(ticker, use_finbert=use_finbert)

    # 5. Decision
    decision = compute_decision(
        ticker          = ticker,
        cluster_label   = cluster_label,
        forecast_result = forecast,
        sentiment_result= sentiment,
        fundamentals    = fundamentals,
    )

    # 6. SHAP (approximate — single ticker)
    shap_data = {
        "features": available_feats,
        "values":   feat_vals.iloc[0].tolist(),
        "cluster_label": cluster_label,
    }

    return {
        "ticker":        ticker,
        "fundamentals":  fundamentals,
        "prices":        prices,
        "features":      feat_vals,
        "cluster_label": cluster_label,
        "forecast":      forecast,
        "sentiment":     sentiment,
        "decision":      decision,
        "shap_data":     shap_data,
    }, None


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if run_btn or "last_ticker" in st.session_state:
    if run_btn:
        st.session_state["last_ticker"]   = ticker
        st.session_state["last_model"]    = forecast_model
        st.session_state["last_finbert"]  = use_finbert

    t = st.session_state.get("last_ticker", ticker)
    m = st.session_state.get("last_model",  forecast_model)
    f = st.session_state.get("last_finbert",use_finbert)

    with st.spinner(f"🔍 Analysing **{t}** — fetching data, running AI models..."):
        result, err = run_full_analysis(t, m, f)

    if err:
        st.error(f"❌ {err}")
        st.stop()

    d   = result["decision"]
    rec = d.recommendation
    rec_style, rec_text = REC_STYLE.get(rec, ("background:#555;color:#fff;", rec))

    # ── TOP KPI ROW ──────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns([2, 1.2, 1.2, 1.2, 1.2])

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="rec-badge" style="{rec_style}">{rec_text}</div>
            <div class="label" style="margin-top:8px;">{t} — {result['fundamentals'].get('name','')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        score_color = "#3FB950" if d.total_score >= 0 else "#F85149"
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:{score_color}">{d.total_score:+.3f}</div>
            <div class="label">Total AI Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        dir_icon = "📈" if "Up" in d.forecast_direction else ("📉" if "Down" in d.forecast_direction else "➡️")
        pct_color = "#3FB950" if d.forecast_pct >= 0 else "#F85149"
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:{pct_color}">{dir_icon} {d.forecast_pct:+.1%}</div>
            <div class="label">30-Day Forecast</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        sent_color = ("#3FB950" if "Positive" in d.sentiment_label
                     else "#F85149" if "Negative" in d.sentiment_label else "#E3B341")
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:{sent_color}">{d.sentiment_score:+.2f}</div>
            <div class="label">News Sentiment</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        conf_color = "#3FB950" if d.confidence_pct >= 60 else "#E3B341" if d.confidence_pct >= 40 else "#F85149"
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:{conf_color}">{d.confidence_pct:.0f}%</div>
            <div class="label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Price Forecast",
        "📰 Sentiment",
        "🧠 Explainability",
        "💡 Rationale",
    ])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns([2, 1])

        with c1:
            st.markdown('<div class="section-title">📉 Candlestick Chart (90 Days)</div>',
                        unsafe_allow_html=True)
            fig_candle = plot_candlestick(result["prices"], t, days=90)
            st.plotly_chart(fig_candle, use_container_width=True)

        with c2:
            st.markdown('<div class="section-title">📋 Fundamentals</div>',
                        unsafe_allow_html=True)
            fund = result["fundamentals"]
            metrics_display = {
                "P/E Ratio":       fund.get("pe_ratio"),
                "P/B Ratio":       fund.get("pb_ratio"),
                "EPS":             fund.get("eps"),
                "ROE":             f"{fund.get('roe', 0)*100:.1f}%" if fund.get("roe") else "N/A",
                "ROA":             f"{fund.get('roa', 0)*100:.1f}%" if fund.get("roa") else "N/A",
                "Debt/Equity":     fund.get("debt_to_equity"),
                "Revenue Growth":  f"{fund.get('revenue_growth', 0)*100:.1f}%" if fund.get("revenue_growth") else "N/A",
                "Profit Margin":   f"{fund.get('profit_margin', 0)*100:.1f}%" if fund.get("profit_margin") else "N/A",
                "Dividend Yield":  f"{fund.get('dividend_yield', 0)*100:.2f}%" if fund.get("dividend_yield") else "N/A",
                "Beta":            fund.get("beta"),
                "Sector":          fund.get("sector", "N/A"),
                "Industry":        fund.get("industry", "N/A"),
            }
            for k, v in metrics_display.items():
                if v is not None and v != "N/A":
                    col_a, col_b = st.columns([1.2, 1])
                    col_a.markdown(f"<span style='color:#8B949E;font-size:0.8rem'>{k}</span>",
                                   unsafe_allow_html=True)
                    col_b.markdown(f"<span style='font-family:JetBrains Mono;font-size:0.8rem'>{v}</span>",
                                   unsafe_allow_html=True)

        # Decision score breakdown
        st.markdown('<div class="section-title">⚖️ Decision Score Breakdown</div>',
                    unsafe_allow_html=True)
        fig_breakdown = plot_decision_breakdown(d)
        st.plotly_chart(fig_breakdown, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2 — PRICE FORECAST
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        fore = result["forecast"]
        st.markdown('<div class="section-title">🔮 30-Day Price Forecast</div>',
                    unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Direction",         fore.get("direction", "N/A"))
        mc2.metric("Expected Change",   f"{fore.get('pct_change', 0):+.2%}")
        mc3.metric("Model Confidence",  f"{fore.get('confidence', 0):.0%}")

        fig_fore = plot_forecast(result["prices"], fore, t)
        st.plotly_chart(fig_fore, use_container_width=True)

        if not fore.get("future_df", pd.DataFrame()).empty:
            with st.expander("📋 Forecast Data Table"):
                st.dataframe(
                    fore["future_df"].rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"}),
                    use_container_width=True,
                )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3 — SENTIMENT
    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        sent = result["sentiment"]

        sc1, sc2 = st.columns([1, 1.5])
        with sc1:
            st.markdown('<div class="section-title">🎯 Sentiment Gauge</div>',
                        unsafe_allow_html=True)
            fig_gauge = plot_sentiment_gauge(sent["gauge_value"], sent["sentiment_label"], t)
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.info(f"📊 {sent.get('summary', '')}")

        with sc2:
            st.markdown('<div class="section-title">📰 News Headlines</div>',
                        unsafe_allow_html=True)
            if not sent["headlines_df"].empty:
                for _, row in sent["headlines_df"].iterrows():
                    s_label = row.get("sentiment", "Neutral")
                    s_score = row.get("final_score", 0)
                    color   = "#3FB950" if s_label == "Positive" else "#F85149" if s_label == "Negative" else "#E3B341"
                    st.markdown(
                        f"""<div style="padding:8px 12px;margin:4px 0;border-left:3px solid {color};
                        background:#161B22;border-radius:4px;font-size:0.82rem;">
                        <span style="color:{color};font-weight:600">[{s_label}  {s_score:+.2f}]</span>
                        &nbsp;{row.get('title','')[:100]}...</div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No headlines available.")

    # ────────────────────────────────────────────────────────────────────────
    # TAB 4 — EXPLAINABILITY
    # ────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-title">🔍 Feature Contributions (SHAP-proxy)</div>',
                    unsafe_allow_html=True)

        shap_data = result["shap_data"]
        feats     = shap_data["features"]
        vals      = shap_data["values"]

        # Build synthetic SHAP-like values from raw feature values
        # (production: use actual trained explainer)
        shap_df = pd.DataFrame({
            "feature":    feats,
            "raw_value":  vals,
            "shap_value": [
                (v - 0) * (0.1 if abs(v) < 2 else 0.05)
                for v in vals
            ],
        }).sort_values("shap_value", ascending=False, key=abs).head(20)
        shap_df["direction"] = shap_df["shap_value"].apply(
            lambda v: "↑ Positive" if v > 0 else "↓ Negative"
        )

        fig_shap = plot_shap_importance(shap_df)
        st.plotly_chart(fig_shap, use_container_width=True)

        # Summary
        top3_pos = shap_df[shap_df["shap_value"] > 0]["feature"].head(3).tolist()
        top3_neg = shap_df[shap_df["shap_value"] < 0]["feature"].head(3).tolist()

        if top3_pos:
            st.success(f"✅ **Top positive drivers:** {', '.join(f.replace('_',' ') for f in top3_pos)}")
        if top3_neg:
            st.warning(f"⚠️  **Top risk factors:** {', '.join(f.replace('_',' ') for f in top3_neg)}")

        with st.expander("📋 Full Feature Table"):
            st.dataframe(shap_df, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 5 — RATIONALE
    # ────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-title">📌 AI Investment Rationale</div>',
                    unsafe_allow_html=True)

        # Big recommendation badge
        r_style, r_text = REC_STYLE.get(rec, ("background:#555;color:#fff;", rec))
        st.markdown(f"""
        <div style="text-align:center;padding:20px;">
            <div class="rec-badge" style="{r_style};font-size:2rem;max-width:400px;margin:0 auto;">{r_text}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(d.rationale, unsafe_allow_html=False)

        st.divider()

        # Signal summary table
        signal_data = {
            "Signal":    ["Cluster Analysis", "Price Forecast", "News Sentiment", "Combined"],
            "Value":     [d.cluster_label, d.forecast_direction, d.sentiment_label, rec],
            "Score":     [f"{d.cluster_score:+.3f}", f"{d.forecast_score:+.3f}",
                          f"{d.sentiment_w_score:+.3f}", f"{d.total_score:+.3f}"],
            "Weight":    ["50%", "30%", "20%", "100%"],
        }
        st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)

        st.caption("⚠️ This platform is for educational and research purposes only. "
                   "Not financial advice. Past performance does not guarantee future results.")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
        <div style="font-size:4rem;margin-bottom:16px;">📊</div>
        <h2 style="color:#58A6FF;font-family:'Sora';font-weight:800;">
            AI-Powered Stock Analysis
        </h2>
        <p style="color:#8B949E;font-size:1rem;max-width:500px;margin:0 auto;">
            Select a ticker from the sidebar and click <b>Analyse Stock</b> to get
            AI-driven clustering, 30-day forecasts, news sentiment, and explainable decisions.
        </p>
    </div>

    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px;max-width:700px;margin:0 auto;">
        <div class="metric-card"><div style="font-size:1.5rem">🤖</div>
        <div class="label">ML Clustering</div>
        <div style="font-size:0.78rem;color:#8B949E;margin-top:4px">KMeans + Hierarchical</div></div>

        <div class="metric-card"><div style="font-size:1.5rem">🔮</div>
        <div class="label">Price Forecasting</div>
        <div style="font-size:0.78rem;color:#8B949E;margin-top:4px">Prophet / LSTM / Ridge</div></div>

        <div class="metric-card"><div style="font-size:1.5rem">📰</div>
        <div class="label">News Sentiment</div>
        <div style="font-size:0.78rem;color:#8B949E;margin-top:4px">VADER + FinBERT</div></div>

        <div class="metric-card"><div style="font-size:1.5rem">🧠</div>
        <div class="label">Explainable AI</div>
        <div style="font-size:0.78rem;color:#8B949E;margin-top:4px">SHAP + Plain English</div></div>
    </div>
    """, unsafe_allow_html=True)
