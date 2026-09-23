"""Data fetching + scoring. No UI code here."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

POPULAR_STOCKS = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Alphabet (Google)": "GOOGL",
    "Amazon": "AMZN", "Meta": "META", "Tesla": "TSLA", "Berkshire Hathaway": "BRK-B",
    "JPMorgan": "JPM", "Johnson & Johnson": "JNJ", "Coca-Cola": "KO", "Walmart": "WMT",
    "LVMH": "MC.PA", "TotalEnergies": "TTE.PA", "Airbus": "AIR.PA",
}
POPULAR_ETFS = {
    "S&P 500 (SPY)": "SPY", "S&P 500 (VOO)": "VOO", "Nasdaq 100 (QQQ)": "QQQ",
    "Total US Market (VTI)": "VTI", "World (VT)": "VT", "MSCI World (URTH)": "URTH",
    "Emerging Markets (VWO)": "VWO", "Bonds (BND)": "BND", "Gold (GLD)": "GLD",
    "Dividend (SCHD)": "SCHD",
}


@dataclass(frozen=True)
class Metric:
    key: str      # key in the `values` dict
    label: str    # must match a glossary entry name
    kind: str     # "low" = lower is better (must be > 0), "low0" = lower is better (0 ok), "high" = higher is better
    good: float
    fair: float
    fmt: str      # "x", "pct", "pct_raw", "big"

    def rule_text(self) -> str:
        if self.kind == "high":
            return f"≥ {self.good:g} good · ≥ {self.fair:g} fair"
        return f"≤ {self.good:g} good · ≤ {self.fair:g} fair"


# Thresholds are classic rules of thumb, NOT universal truths (they vary by sector).
STOCK_METRICS = [
    Metric("trailingPE", "P/E Ratio", "low", 15, 25, "x"),
    Metric("forwardPE", "Forward P/E", "low", 15, 22, "x"),
    Metric("pegRatio", "PEG Ratio", "low", 1, 2, "x"),
    Metric("priceToSalesTrailing12Months", "P/S Ratio", "low", 2, 5, "x"),
    Metric("priceToBook", "P/B Ratio", "low", 3, 6, "x"),
    Metric("enterpriseToEbitda", "EV/EBITDA", "low", 10, 15, "x"),
    Metric("returnOnEquity", "ROE", "high", 0.15, 0.08, "pct"),
    Metric("profitMargins", "Profit Margin", "high", 0.15, 0.05, "pct"),
    Metric("revenueGrowth", "Revenue Growth", "high", 0.10, 0.03, "pct"),
    Metric("debtToEquity", "Debt/Equity", "low0", 0.5, 1.0, "x"),
    Metric("currentRatio", "Current Ratio", "high", 1.5, 1.0, "x"),
]
ETF_METRICS = [
    Metric("expense_pct", "Expense Ratio", "low0", 0.20, 0.50, "pct_raw"),
    Metric("totalAssets", "Fund Size (AUM)", "high", 1e9, 1e8, "big"),
]
PRICE_METRICS = [  # computed by us from price history
    Metric("ret_1y", "1Y Return", "high", 0.10, 0.0, "pct"),
    Metric("vol", "Volatility", "low0", 0.20, 0.35, "pct"),
    Metric("mdd", "Max Drawdown", "high", -0.20, -0.35, "pct"),
]

POINTS = {"good": 1.0, "fair": 0.5, "bad": 0.0}
EMOJI = {"good": "🟢 Good", "fair": "🟡 Fair", "bad": "🔴 Weak", None: "⚪ n/a"}


def is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def grade(m: Metric, v):
    if is_missing(v):
        return None
    if m.kind == "low" and v <= 0:  # negative P/E etc. = losing money
        return "bad"
    if m.kind in ("low", "low0"):
        return "good" if v <= m.good else "fair" if v <= m.fair else "bad"
    return "good" if v >= m.good else "fair" if v >= m.fair else "bad"


def fmt(v, f: str) -> str:
    if is_missing(v):
        return "n/a"
    if f == "pct":
        return f"{v * 100:.1f}%"
    if f == "pct_raw":
        return f"{v:.2f}%"
    if f == "big":
        for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
            if abs(v) >= div:
                return f"{v / div:.1f}{unit}"
        return f"{v:,.0f}"
    return f"{v:.2f}"


@st.cache_data(ttl=900, show_spinner=False)
def fetch(ticker: str):
    t = yf.Ticker(ticker)
    hist = t.history(period="5y", auto_adjust=True)
    info = t.info or {}
    return info, hist


def price_stats(hist: pd.DataFrame) -> dict:
    close = hist["Close"].dropna().tail(252)
    if len(close) < 30:
        return {}
    daily = close.pct_change().dropna()
    return {
        "ret_1y": close.iloc[-1] / close.iloc[0] - 1,
        "vol": daily.std() * np.sqrt(252),
        "mdd": (close / close.cummax() - 1).min(),
    }


def normalize(info: dict) -> dict:
    v = dict(info)
    if v.get("debtToEquity") is not None:  # Yahoo gives 150 for 1.5x
        v["debtToEquity"] = v["debtToEquity"] / 100
    exp = v.get("netExpenseRatio")
    if exp is None and v.get("annualReportExpenseRatio") is not None:
        exp = v["annualReportExpenseRatio"] * 100
    v["expense_pct"] = exp
    return v


def analyze(ticker: str) -> dict:
    info, hist = fetch(ticker)
    if hist.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the symbol.")
    values = {**normalize(info), **price_stats(hist)}
    is_fund = info.get("quoteType") in ("ETF", "MUTUALFUND")
    metrics = (ETF_METRICS if is_fund else STOCK_METRICS) + PRICE_METRICS

    rows, pts = [], []
    for m in metrics:
        v = values.get(m.key)
        g = grade(m, v)
        if g:
            pts.append(POINTS[g])
        rows.append({"Measure": m.label, "Value": fmt(v, m.fmt), "Verdict": EMOJI[g], "Rule of thumb": m.rule_text()})

    score = round(100 * sum(pts) / len(pts)) if pts else None
    return {
        "ticker": ticker.upper(), "name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "is_fund": is_fund, "info": info, "hist": hist, "rows": rows, "score": score,
    }


def verdict(score):
    if score is None:
        return "Not enough data", "gray"
    if score >= 70:
        return "Looks attractive", "green"
    if score >= 45:
        return "Mixed / fairly valued", "orange"
    return "Looks expensive or weak", "red"


def summary_text(a: dict) -> str:
    """Plain-text snapshot handed to the AI as context."""
    info = a["info"]
    lines = [
        f"{a['name']} ({a['ticker']}) - {'ETF/fund' if a['is_fund'] else 'stock'}",
        f"Sector: {info.get('sector', 'n/a')} | Industry: {info.get('industry', 'n/a')}",
        f"Price: {info.get('currentPrice') or info.get('regularMarketPrice', 'n/a')} {info.get('currency', '')}",
        f"Overall rule-based score: {a['score']}/100 ({verdict(a['score'])[0]})",
        "Measures:",
    ]
    lines += [f"- {r['Measure']}: {r['Value']} -> {r['Verdict']}" for r in a["rows"]]
    return "\n".join(lines)
