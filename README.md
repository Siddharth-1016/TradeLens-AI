# ⚡ QuantAI — AI-Powered Investment Decision Platform

> **Production-grade fintech platform** combining Unsupervised ML, Time-Series Forecasting,
> NLP Sentiment Analysis, and Explainable AI to generate stock investment signals.

---

## 🏗️ Architecture

```
User Input (Ticker)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION LAYER                           │
│   yfinance → OHLCV (5yr) + Fundamentals (PE, ROE, D/E, ...)        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                               │
│   SMA/EMA/RSI/MACD/BB/ATR/Momentum/52W Range + Fundamental Ratios  │
└──────┬──────────────────┬──────────────────────────────────────────┘
       │                  │
       ▼                  ▼
┌──────────────┐  ┌───────────────┐  ┌──────────────────────────────┐
│  CLUSTERING  │  │  FORECASTING  │  │    NEWS SENTIMENT             │
│  KMeans +    │  │  Prophet /    │  │    yfinance RSS →             │
│  Hierarchical│  │  LSTM /Ridge  │  │    VADER + FinBERT            │
│  → BUY /     │  │  → 30d price  │  │    → Score [-1, +1]          │
│    MAYBE/NOT │  │  + direction  │  │    → Positive/Neutral/Neg     │
└──────┬───────┘  └───────┬───────┘  └──────────────┬───────────────┘
       │                  │                           │
       ▼                  ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DECISION ENGINE                                  │
│   Score = 0.50 × Cluster + 0.30 × Forecast + 0.20 × Sentiment      │
│   → STRONG BUY | BUY | HOLD | AVOID                                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│               EXPLAINABILITY (SHAP)                                 │
│   XGBoost Proxy → SHAP values → Feature importance → Plain English │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│            STREAMLIT DASHBOARD  (app.py)                           │
│   Candlestick | Forecast | Sentiment Gauge | SHAP Bar | Radar      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
quant_platform/
├── app.py                          ← Streamlit web application
├── requirements.txt                ← Python dependencies
├── README.md
│
├── src/
│   ├── data_collection.py          ← yfinance OHLCV + fundamentals
│   ├── feature_engineering.py      ← Technical indicators + feature pipeline
│   ├── clustering.py               ← KMeans, Hierarchical, label mapping
│   ├── explainability.py           ← SHAP proxy model + plain-English
│   ├── forecasting.py              ← Prophet / LSTM / Ridge fallback
│   ├── sentiment.py                ← VADER + FinBERT news sentiment
│   ├── decision_engine.py          ← Multi-signal score fusion
│   └── visualization.py            ← All Plotly charts
│
└── data/
    ├── raw/                        ← Raw OHLCV CSVs
    └── processed/                  ← Feature-engineered CSVs
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install requirements
pip install -r requirements.txt

# Download NLTK data (one-time)
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 2. Run the Streamlit App

```bash
cd quant_platform
streamlit run app.py
```

Open → `http://localhost:8501`

### 3. Pre-collect Full Universe (Optional)

```bash
# Collect S&P 500 top 50 (takes ~5 minutes)
python src/data_collection.py --market SP500 --max 50

# Run feature engineering
python src/feature_engineering.py

# Run full clustering on universe
python src/clustering.py
```

---

## 🧠 AI Modules

| Module | Algorithm | Purpose |
|--------|-----------|---------|
| **Clustering** | KMeans + Hierarchical | Group stocks by quality profile |
| **Forecasting** | Prophet (Meta) | 30-day price trend prediction |
| **Sentiment** | VADER + FinBERT | News headline sentiment score |
| **Explainability** | SHAP (XGBoost proxy) | Feature attribution per prediction |
| **Decision** | Weighted score fusion | Final BUY / HOLD / AVOID signal |

---

## 📊 Feature Engineering Pipeline

### Technical Indicators (Why each matters)

| Feature | Why It Matters |
|---------|----------------|
| SMA 20/50/200 | Trend direction at 3 time horizons |
| EMA 12/26 | Recency-weighted trend, base of MACD |
| RSI | Overbought/oversold momentum oscillator |
| MACD | Trend + momentum convergence signal |
| Bollinger %B | Price position within volatility band |
| ATR | True volatility for risk sizing |
| Momentum 1M/3M/6M | Rate of price change / trend strength |
| 52W High/Low % | Relative strength vs annual range |
| Volume Z-score | Institutional conviction / accumulation |

### Fundamental Metrics

| Metric | Signal |
|--------|--------|
| ROE | Management efficiency of equity capital |
| D/E Ratio | Financial risk / leverage |
| Revenue Growth | Top-line momentum |
| Profit Margin | Operational quality |
| P/E Ratio | Valuation expensive or cheap |

---

## 🎯 Cluster Label Logic

```
Cluster Quality Score = Σ(positive features) - Σ(risk features)

Positive: ROE, ROA, momentum_3m, revenue_growth, profit_margin
Negative: debt_to_equity, volatility_20d

→ Score Rank 1  =  🟢 BUY
→ Score Rank 2  =  🟡 MAYBE BUY
→ Score Rank 3  =  🔴 NOT BUY
```

---

## ⚖️ Decision Formula

```python
Score = 0.50 × ClusterScore      # Fundamental + technical quality
      + 0.30 × ForecastScore     # Prophet direction × confidence
      + 0.20 × SentimentScore    # News sentiment magnitude

# Thresholds:
Score ≥  0.55  →  🚀 STRONG BUY
Score ≥  0.15  →  🟢 BUY
Score ≥ -0.15  →  🟡 HOLD
Score <  -0.15  →  🔴 AVOID
```

---

## ☁️ Deployment

### Streamlit Cloud (Free)
1. Push repo to GitHub
2. Go to https://share.streamlit.io
3. Connect repo → set `app.py` as entry point
4. Deploy ✅

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t quantai .
docker run -p 8501:8501 quantai
```

### AWS / GCP / Azure
Deploy the Docker container to any cloud container service (ECS, Cloud Run, ACI).

---

## ⚠️ Disclaimer

This platform is for **educational and research purposes only**.
It does not constitute financial advice. Always consult a qualified
financial advisor before making investment decisions.

---

## 📜 License

MIT License — free to use, modify, and distribute.
#   T r a d e L e n s - A I  
 