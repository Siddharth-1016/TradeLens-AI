# ⚡ TradeLens-AI

> **An AI-Powered Investment Decision Platform** 
> Combines Machine Learning, Time-Series Forecasting, NLP Sentiment Analysis, and Explainable AI to generate clear stock investment signals.

---

## ✨ Key Features

*   **Clustering (Quality Profiling):** Uses KMeans and Hierarchical clustering to group stocks based on technical and fundamental health.
*   **Time-Series Forecasting:** Leverages Meta's Prophet model to predict 30-day price trends.
*   **News Sentiment Analysis:** Scrapes financial news and uses VADER + FinBERT to gauge market sentiment.
*   **Explainable AI (SHAP):** Breaks down the "why" behind every buy/sell signal in plain English.
*   **Interactive Dashboard:** A complete, user-friendly Streamlit web interface.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required once)
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 2. Run the Dashboard
```bash
cd quant_platform
streamlit run app.py
```
*Your app will open automatically at `http://localhost:8501`*

---

## 📁 Core Project Structure

```text
quant_platform/
├── app.py                       ← Main Streamlit dashboard
├── requirements.txt             ← Project dependencies
├── src/
│   ├── data_collection.py       ← Fetches market data (yfinance)
│   ├── feature_engineering.py   ← Calculates technical indicators
│   ├── forecasting.py           ← Price prediction models
│   ├── sentiment.py             ← News sentiment analysis
│   └── decision_engine.py       ← Final buy/hold/avoid logic
└── data/                        ← Local data storage
```

---

## ☁️ Quick Deployment (Docker)

To run TradeLens-AI in an isolated container:

```bash
docker build -t tradelens-ai .
docker run -p 8501:8501 tradelens-ai
```

---

## ⚠️ Disclaimer

This platform is for **educational and research purposes only**. It does not constitute financial advice. Always consult a qualified financial advisor before making actual investment decisions.

---

## 📜 License

**MIT License** — free to use, modify, and distribute.