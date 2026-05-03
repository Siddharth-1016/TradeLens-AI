"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 9 — DECISION ENGINE                                   ║
║  Combines Cluster + Forecast + Sentiment → Final Signal     ║
╚══════════════════════════════════════════════════════════════╝

DECISION FORMULA:
─────────────────────────────────────────────────────────────────
    Score = w1 × ClusterScore
          + w2 × ForecastScore
          + w3 × SentimentScore

    w1 = 0.50  (Cluster is most reliable — fundamental + technical)
    w2 = 0.30  (Forecast adds medium-term directional view)
    w3 = 0.20  (Sentiment adds near-term news catalyst signal)

THRESHOLDS → FINAL RECOMMENDATION:
    Score ≥  0.60  → STRONG BUY  🚀
    Score ≥  0.20  → BUY         🟢
    Score ≥ -0.20  → HOLD        🟡
    Score <  -0.20 → AVOID       🔴
─────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# SCORE ENCODING MAPS
# ══════════════════════════════════════════════════════════════════════════════

CLUSTER_SCORE_MAP = {
    "🟢 BUY":       +1.0,
    "🟡 MAYBE BUY": +0.0,
    "🔴 NOT BUY":   -1.0,
}

FORECAST_SCORE_MAP = {
    "📈 Up":        +1.0,
    "➡️ Sideways":  +0.0,
    "📉 Down":      -1.0,
}

SENTIMENT_SCORE_MAP = {
    "🟢 Positive":  +1.0,
    "🟡 Neutral":    0.0,
    "🔴 Negative":  -1.0,
}

# Weights (must sum to 1)
W_CLUSTER   = 0.50
W_FORECAST  = 0.30
W_SENTIMENT = 0.20


# ══════════════════════════════════════════════════════════════════════════════
# DECISION DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class InvestmentDecision:
    ticker:          str
    company_name:    str
    sector:          str

    # Component signals
    cluster_label:   str
    forecast_direction: str
    forecast_pct:    float
    sentiment_label: str
    sentiment_score: float

    # Weights and scores
    cluster_score:   float
    forecast_score:  float
    sentiment_w_score: float
    total_score:     float

    # Final output
    recommendation:  str
    confidence_pct:  float
    rationale:       str

    # SHAP explanation (optional)
    shap_explanation: str = ""

    # Forecast confidence
    forecast_confidence: float = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def normalise_score(raw: float, confidence: float, default_weight: float = 1.0) -> float:
    """
    Scale a raw signal score by confidence.
    Low-confidence signals contribute less to the final score.
    """
    return raw * confidence * default_weight


def compute_decision(
    ticker:             str,
    cluster_label:      str,
    forecast_result:    dict,
    sentiment_result:   dict,
    fundamentals:       dict = None,
    shap_explanation:   str  = "",
) -> InvestmentDecision:
    """
    Core decision function.

    Parameters
    ----------
    cluster_label    : "🟢 BUY" / "🟡 MAYBE BUY" / "🔴 NOT BUY"
    forecast_result  : dict from forecasting.run_forecast()
    sentiment_result : dict from sentiment.run_sentiment_analysis()
    fundamentals     : dict from data_collection.fetch_fundamentals()
    """
    fundamentals   = fundamentals or {}
    company_name   = fundamentals.get("name",     ticker)
    sector         = fundamentals.get("sector",   "Unknown")

    # ── 1. Cluster Score ─────────────────────────────────────────────────────
    cluster_raw    = CLUSTER_SCORE_MAP.get(cluster_label, 0.0)
    c_score        = cluster_raw * W_CLUSTER

    # ── 2. Forecast Score ─────────────────────────────────────────────────────
    direction      = forecast_result.get("direction",   "➡️ Sideways")
    f_raw          = FORECAST_SCORE_MAP.get(direction,  0.0)
    f_confidence   = forecast_result.get("confidence",  0.5)
    pct_change     = forecast_result.get("pct_change",  0.0)

    # Boost signal if expected move is large (> 5%)
    magnitude_boost = min(abs(pct_change) / 0.05, 1.0)
    f_score        = f_raw * f_confidence * magnitude_boost * W_FORECAST

    # ── 3. Sentiment Score ────────────────────────────────────────────────────
    sent_label     = sentiment_result.get("sentiment_label", "🟡 Neutral")
    sent_raw       = SENTIMENT_SCORE_MAP.get(sent_label,     0.0)
    sent_score_abs = abs(sentiment_result.get("sentiment_score", 0.0))
    s_score        = sent_raw * min(sent_score_abs * 3, 1.0) * W_SENTIMENT

    # ── 4. Total Score ────────────────────────────────────────────────────────
    total_score    = c_score + f_score + s_score

    # ── 5. Final Recommendation ───────────────────────────────────────────────
    if total_score >= 0.55:
        recommendation = "🚀 STRONG BUY"
    elif total_score >= 0.15:
        recommendation = "🟢 BUY"
    elif total_score >= -0.15:
        recommendation = "🟡 HOLD"
    else:
        recommendation = "🔴 AVOID"

    # ── 6. Confidence ─────────────────────────────────────────────────────────
    # Map score [-1,+1] → confidence 0–100%
    confidence_pct = int(min(100, max(0, (total_score + 1) / 2 * 100)))

    # ── 7. Rationale ─────────────────────────────────────────────────────────
    rationale = _build_rationale(
        ticker, recommendation, cluster_label,
        direction, pct_change, sent_label,
        c_score, f_score, s_score, fundamentals,
    )

    return InvestmentDecision(
        ticker               = ticker,
        company_name         = company_name,
        sector               = sector,
        cluster_label        = cluster_label,
        forecast_direction   = direction,
        forecast_pct         = float(pct_change),
        sentiment_label      = sent_label,
        sentiment_score      = float(sentiment_result.get("sentiment_score", 0)),
        cluster_score        = c_score,
        forecast_score       = f_score,
        sentiment_w_score    = s_score,
        total_score          = total_score,
        recommendation       = recommendation,
        confidence_pct       = float(confidence_pct),
        rationale            = rationale,
        shap_explanation     = shap_explanation,
        forecast_confidence  = f_confidence,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RATIONALE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_rationale(
    ticker:        str,
    recommendation: str,
    cluster_label: str,
    direction:     str,
    pct_change:    float,
    sent_label:    str,
    c_score:       float,
    f_score:       float,
    s_score:       float,
    fundamentals:  dict,
) -> str:
    """Builds a structured plain-English rationale string."""
    rec_clean = recommendation.split(" ", 1)[-1]  # strip emoji

    lines = [
        f"**{ticker} — Final Recommendation: {rec_clean}**",
        "",
        f"• **Cluster Analysis ({c_score:+.3f}):** Stock is in the `{cluster_label}` cluster "
        f"based on its fundamental and technical profile.",
        f"• **Price Forecast ({f_score:+.3f}):** Model predicts `{direction}` over the next "
        f"30 trading days (expected move: `{pct_change:+.1%}`).",
        f"• **News Sentiment ({s_score:+.3f}):** Recent news is `{sent_label}` "
        f"for {ticker}.",
        "",
    ]

    # Fundamental highlights
    roe   = fundamentals.get("roe")
    d_e   = fundamentals.get("debt_to_equity")
    rev_g = fundamentals.get("revenue_growth")

    if roe and not (isinstance(roe, float) and np.isnan(roe)):
        lines.append(f"• ROE: `{roe:.1%}` | " +
                     ("Strong equity efficiency." if roe > 0.15 else "Below average."))
    if d_e and not (isinstance(d_e, float) and np.isnan(d_e)):
        lines.append(f"• Debt/Equity: `{d_e:.2f}` | " +
                     ("Conservative balance sheet." if d_e < 1.0 else "Elevated leverage."))
    if rev_g and not (isinstance(rev_g, float) and np.isnan(rev_g)):
        lines.append(f"• Revenue Growth: `{rev_g:.1%}` | " +
                     ("Accelerating top-line." if rev_g > 0.10 else "Slow growth."))

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH DECISIONS  (for full universe)
# ══════════════════════════════════════════════════════════════════════════════

def batch_decisions(
    labelled_df:      pd.DataFrame,
    forecast_map:     dict,   # ticker → forecast_result
    sentiment_map:    dict,   # ticker → sentiment_result
    fundamentals_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute decisions for the full stock universe.
    Returns a leaderboard DataFrame.
    """
    records = []
    fund_idx = fundamentals_df.set_index("ticker") if "ticker" in fundamentals_df.columns else fundamentals_df

    for ticker, row in labelled_df.iterrows():
        fund   = fund_idx.loc[ticker].to_dict() if ticker in fund_idx.index else {}
        fore   = forecast_map.get(ticker,  {"direction": "➡️ Sideways", "confidence": 0.5, "pct_change": 0.0})
        sent   = sentiment_map.get(ticker, {"sentiment_label": "🟡 Neutral", "sentiment_score": 0.0})
        clabel = row.get("investment_label", "🟡 MAYBE BUY")

        decision = compute_decision(ticker, clabel, fore, sent, fund)
        records.append({
            "ticker":          decision.ticker,
            "company":         decision.company_name,
            "sector":          decision.sector,
            "cluster":         decision.cluster_label,
            "forecast":        decision.forecast_direction,
            "forecast_pct":    f"{decision.forecast_pct:+.1%}",
            "sentiment":       decision.sentiment_label,
            "total_score":     round(decision.total_score, 4),
            "recommendation":  decision.recommendation,
            "confidence":      f"{decision.confidence_pct:.0f}%",
        })

    result = pd.DataFrame(records).sort_values("total_score", ascending=False)
    return result


if __name__ == "__main__":
    # Quick demo
    mock_forecast  = {"direction": "📈 Up", "confidence": 0.72, "pct_change": 0.08}
    mock_sentiment = {"sentiment_label": "🟢 Positive", "sentiment_score": 0.42}
    mock_fund      = {"name": "Apple Inc.", "sector": "Technology",
                      "roe": 0.18, "debt_to_equity": 0.9, "revenue_growth": 0.12}

    decision = compute_decision(
        ticker          = "AAPL",
        cluster_label   = "🟢 BUY",
        forecast_result = mock_forecast,
        sentiment_result= mock_sentiment,
        fundamentals    = mock_fund,
    )

    print(f"\n{'='*60}")
    print(f"  TICKER:         {decision.ticker}")
    print(f"  RECOMMENDATION: {decision.recommendation}")
    print(f"  TOTAL SCORE:    {decision.total_score:+.4f}")
    print(f"  CONFIDENCE:     {decision.confidence_pct:.0f}%")
    print(f"{'='*60}")
    print(f"\nRationale:\n{decision.rationale}")
