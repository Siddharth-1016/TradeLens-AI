"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 2 — DATA COLLECTION PIPELINE                          ║
║  Collects OHLCV + Fundamentals for NIFTY50 / S&P500         ║
╚══════════════════════════════════════════════════════════════╝

WHY THIS DESIGN:
- yfinance provides free, reliable historical + fundamental data
- 5-year window captures full market cycles (bull/bear/recovery)
- Fundamentals give VALUATION context; technicals give MOMENTUM
- Saving to CSV ensures reproducibility and offline dev
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

# ── Ticker Universes ──────────────────────────────────────────────────────────
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH",  "XOM",  "JNJ",  "JPM",  "V",    "PG",   "MA",   "HD",
    "CVX",  "MRK",  "ABBV", "PFE",  "AVGO", "COST", "KO",   "PEP",
    "LLY",  "WMT",  "MCD",  "CSCO", "TMO",  "ACN",  "ABT",  "CRM",
    "NFLX", "BAC",  "ADBE", "NKE",  "DIS",  "INTC", "VZ",   "CMCSA",
    "TXN",  "NEE",  "RTX",  "QCOM", "UPS",  "PM",   "HON",  "AMD",
    "AMGN", "ORCL",
]

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS",     "HDFCBANK.NS", "INFY.NS",    "HINDUNILVR.NS",
    "ICICIBANK.NS","KOTAKBANK.NS","SBIN.NS",     "BHARTIARTL.NS","ITC.NS",
    "BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS",  "AXISBANK.NS","ULTRACEMCO.NS",
    "TITAN.NS",    "WIPRO.NS",   "HCLTECH.NS",  "SUNPHARMA.NS","TATAMOTORS.NS",
    "NESTLEIND.NS","POWERGRID.NS","TECHM.NS",    "NTPC.NS",    "BAJAJFINSV.NS",
    "ONGC.NS",     "GRASIM.NS",  "JSWSTEEL.NS", "LT.NS",      "TATASTEEL.NS",
    "COALINDIA.NS","CIPLA.NS",   "DIVISLAB.NS", "EICHERMOT.NS","ADANIPORTS.NS",
    "INDUSINDBK.NS","DRREDDY.NS","SBILIFE.NS",  "BPCL.NS",    "BRITANNIA.NS",
    "HDFCLIFE.NS", "TATACONSUM.NS","M&M.NS",    "APOLLOHOSP.NS","HINDALCO.NS",
    "BAJAJ-AUTO.NS","UPL.NS",    "SHREECEM.NS", "HEROMOTOCO.NS","ADANIENT.NS",
]


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL METRICS EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
def fetch_fundamentals(ticker: str) -> dict:
    """
    Pull key fundamental metrics from yfinance.
    Returns a flat dict keyed by metric name.

    WHY THESE METRICS:
    - P/E    → valuation vs earnings (high P/E = expensive or high growth)
    - P/B    → price vs book value (< 1 = potentially undervalued)
    - EPS    → absolute earnings power
    - ROE    → management efficiency of equity capital
    - ROA    → asset utilisation efficiency
    - D/E    → financial leverage risk
    - RevGrowth → momentum in top-line
    - ProfitMargin → operational quality
    - MarketCap → liquidity tier
    - DivYield → income factor
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info

        return {
            "ticker":         ticker,
            "pe_ratio":       info.get("trailingPE",          np.nan),
            "pb_ratio":       info.get("priceToBook",         np.nan),
            "eps":            info.get("trailingEps",         np.nan),
            "roe":            info.get("returnOnEquity",      np.nan),
            "roa":            info.get("returnOnAssets",      np.nan),
            "debt_to_equity": info.get("debtToEquity",        np.nan),
            "revenue_growth": info.get("revenueGrowth",       np.nan),
            "profit_margin":  info.get("profitMargins",       np.nan),
            "market_cap":     info.get("marketCap",           np.nan),
            "dividend_yield": info.get("dividendYield",       np.nan),
            "beta":           info.get("beta",                np.nan),
            "sector":         info.get("sector",              "Unknown"),
            "industry":       info.get("industry",            "Unknown"),
            "name":           info.get("shortName",           ticker),
        }
    except Exception as e:
        logger.warning(f"Fundamentals failed for {ticker}: {e}")
        return {"ticker": ticker}


# ══════════════════════════════════════════════════════════════════════════════
# OHLCV PRICE DATA COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════
def fetch_ohlcv(ticker: str, start: str = START_DATE, end: str = END_DATE) -> Optional[pd.DataFrame]:
    """
    Downloads 5-year daily OHLCV data for one ticker.
    Adds adjusted-close log returns for stationarity.
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

        if df.empty or len(df) < 100:
            logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
            return None

        # ⭐ FIX FOR NEW YFINANCE MULTI-INDEX BUG
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # convert to lowercase safely
        df.columns = [str(c).lower() for c in df.columns]

        df["ticker"] = ticker
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["daily_range"] = (df["high"] - df["low"]) / df["close"]

        df.dropna(subset=["close"], inplace=True)
        df.index.name = "date"

        return df

    except Exception as e:
        logger.error(f"OHLCV failed for {ticker}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BATCH COLLECTOR — FULL UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════
def collect_universe(
    tickers: list,
    market:  str = "SP500",
    delay:   float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterates ticker universe, collects fundamentals + OHLCV.

    Returns
    -------
    fundamentals_df : one row per ticker, all fundamental metrics
    prices_df       : long-format OHLCV dataframe for all tickers
    """
    logger.info(f"Starting collection for {len(tickers)} tickers [{market}]")

    fund_records = []
    price_frames = []

    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1:02d}/{len(tickers)}] Collecting {ticker} ...")

        # Fundamentals
        fund = fetch_fundamentals(ticker)
        fund_records.append(fund)

        # OHLCV
        prices = fetch_ohlcv(ticker)
        if prices is not None:
            price_frames.append(prices.reset_index())

        time.sleep(delay)   # be polite to the API

    fundamentals_df = pd.DataFrame(fund_records)
    prices_df       = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()

    # ── Persist ──────────────────────────────────────────────────────────────
    fund_path   = os.path.join(PROCESSED_DIR, f"{market}_fundamentals.csv")
    prices_path = os.path.join(RAW_DIR,       f"{market}_prices.csv")

    fundamentals_df.to_csv(fund_path,   index=False)
    prices_df.to_csv(      prices_path, index=False)

    logger.info(f"Saved fundamentals → {fund_path}")
    logger.info(f"Saved prices       → {prices_path}")
    logger.info(f"Fundamentals shape : {fundamentals_df.shape}")
    logger.info(f"Prices shape       : {prices_df.shape}")

    return fundamentals_df, prices_df


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE TICKER FETCH  (used by the Streamlit app at runtime)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_single_ticker(ticker: str) -> tuple[dict, pd.DataFrame]:
    """
    Lightweight function called by the Streamlit app for on-demand data.
    Returns (fundamentals_dict, ohlcv_dataframe).
    """
    fund   = fetch_fundamentals(ticker)
    prices = fetch_ohlcv(ticker)
    return fund, prices


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stock Data Collector")
    parser.add_argument("--market", choices=["SP500", "NIFTY50"], default="SP500")
    parser.add_argument("--max",    type=int, default=10, help="Limit tickers for quick test")
    args = parser.parse_args()

    tickers = SP500_TICKERS if args.market == "SP500" else NIFTY50_TICKERS
    tickers = tickers[: args.max]

    fund_df, price_df = collect_universe(tickers, market=args.market)
    print("\n=== Fundamentals Sample ===")
    print(fund_df[["ticker", "pe_ratio", "pb_ratio", "roe", "debt_to_equity"]].head())
    print("\n=== Price Data Sample ===")
    print(price_df[["date", "ticker", "close", "volume"]].head())
