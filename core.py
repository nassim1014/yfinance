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
    category: str

    def rule_text(self) -> str:
        if self.kind == "high":
            return f"Good: ≥ {fmt(self.good, self.fmt)} · Fair: ≥ {fmt(self.fair, self.fmt)}"
        return f"Good: ≤ {fmt(self.good, self.fmt)} · Fair: ≤ {fmt(self.fair, self.fmt)}"


# Thresholds are classic rules of thumb, NOT universal truths (they vary by sector).
STOCK_METRICS = [
    Metric("trailingPE", "P/E Ratio", "low", 15, 25, "x", "Valuation"),
    Metric("forwardPE", "Forward P/E", "low", 15, 22, "x", "Valuation"),
    Metric("pegRatio", "PEG Ratio", "low", 1, 2, "x", "Valuation"),
    Metric("priceToSalesTrailing12Months", "P/S Ratio", "low", 2, 5, "x", "Valuation"),
    Metric("priceToBook", "P/B Ratio", "low", 3, 6, "x", "Valuation"),
    Metric("enterpriseToEbitda", "EV/EBITDA", "low", 10, 15, "x", "Valuation"),
    Metric("returnOnEquity", "ROE", "high", 0.15, 0.08, "pct", "Business quality"),
    Metric("profitMargins", "Profit Margin", "high", 0.15, 0.05, "pct", "Business quality"),
    Metric("revenueGrowth", "Revenue Growth", "high", 0.10, 0.03, "pct", "Business quality"),
    Metric("debtToEquity", "Debt/Equity", "low0", 0.5, 1.0, "x", "Financial strength"),
    Metric("currentRatio", "Current Ratio", "high", 1.5, 1.0, "x", "Financial strength"),
]
ETF_METRICS = [
    Metric("expense_pct", "Expense Ratio", "low0", 0.20, 0.50, "pct_raw", "Costs"),
    Metric("totalAssets", "Fund Size (AUM)", "high", 1e9, 1e8, "big", "Fund quality"),
]
PRICE_METRICS = [  # computed by us from price history
    Metric("ret_1y", "1Y Return", "high", 0.10, 0.0, "pct", "Market return & risk"),
    Metric("vol", "Volatility", "low0", 0.20, 0.35, "pct", "Market return & risk"),
    Metric("mdd", "Max Drawdown", "high", -0.20, -0.35, "pct", "Market return & risk"),
]

STOCK_CATEGORY_WEIGHTS = {
    "Valuation": 0.30,
    "Business quality": 0.30,
    "Financial strength": 0.20,
    "Market return & risk": 0.20,
}
ETF_CATEGORY_WEIGHTS = {
    "Costs": 0.35,
    "Fund quality": 0.25,
    "Market return & risk": 0.40,
}

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


def grade_explanation(m: Metric, v, g: str | None) -> str:
    """Explain a metric's rule-based grade in the language shown in the UI."""
    if g is None:
        return "No value was available, so this measure does not affect the score."

    value, good, fair = fmt(v, m.fmt), fmt(m.good, m.fmt), fmt(m.fair, m.fmt)
    if m.kind == "low" and v <= 0:
        return f"Weak: {value} is zero or negative, which usually means the company is unprofitable."

    direction = "at least" if m.kind == "high" else "at most"
    if g == "good":
        return f"Good: {value} meets the good threshold of {direction} {good}."
    if g == "fair":
        return f"Fair: {value} misses the good threshold but meets the fair threshold of {direction} {fair}."
    return f"Weak: {value} misses the fair threshold of {direction} {fair}."


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


def score_metrics(metrics: list[Metric], values: dict, category_weights: dict) -> tuple:
    """Return an explainable, category-weighted score and the rows used to make it."""
    graded = []
    for m in metrics:
        value = values.get(m.key)
        g = grade(m, value)
        graded.append({"metric": m, "value": value, "grade": g, "points": POINTS.get(g)})

    available = {
        category: [item for item in graded if item["metric"].category == category and item["grade"]]
        for category in category_weights
    }
    active_weights = {
        category: weight for category, weight in category_weights.items() if available[category]
    }
    weight_total = sum(active_weights.values())
    normalized_weights = {
        category: weight / weight_total for category, weight in active_weights.items()
    }

    category_scores = []
    for category, base_weight in category_weights.items():
        items = available[category]
        total_in_category = sum(item["metric"].category == category for item in graded)
        if not items:
            category_scores.append({
                "Category": category,
                "Score": "n/a",
                "Weight": f"{base_weight:.0%} (excluded)",
                "Contribution": "n/a",
                "Coverage": f"0/{total_in_category} measures",
            })
            continue

        category_score = 100 * sum(item["points"] for item in items) / len(items)
        effective_weight = normalized_weights[category]
        category_scores.append({
            "Category": category,
            "Score": f"{category_score:.0f}/100",
            "Weight": f"{effective_weight:.0%}",
            "Contribution": f"{category_score * effective_weight:.1f} points",
            "Coverage": f"{len(items)}/{total_in_category} measures",
        })

    score = (
        round(sum(
            100 * sum(item["points"] for item in available[category]) / len(available[category])
            * normalized_weights[category]
            for category in normalized_weights
        ))
        if normalized_weights else None
    )

    rows = []
    for item in graded:
        m, value, g, points = item["metric"], item["value"], item["grade"], item["points"]
        category_items = available[m.category]
        metric_weight = (
            normalized_weights[m.category] / len(category_items)
            if g and category_items else None
        )
        rows.append({
            "Category": m.category,
            "Measure": m.label,
            "Value": fmt(value, m.fmt),
            "Verdict": EMOJI[g],
            "Metric score": f"{points * 100:.0f}/100" if points is not None else "n/a",
            "Weight": f"{metric_weight:.1%}" if metric_weight is not None else "excluded",
            "Contribution": f"{points * metric_weight * 100:.1f} points" if metric_weight is not None else "n/a",
            "Why": grade_explanation(m, value, g),
            "Rule of thumb": m.rule_text(),
        })
    return score, rows, category_scores


def analyze(ticker: str) -> dict:
    info, hist = fetch(ticker)
    if hist.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the symbol.")
    values = {**normalize(info), **price_stats(hist)}
    is_fund = info.get("quoteType") in ("ETF", "MUTUALFUND")
    metrics = (ETF_METRICS if is_fund else STOCK_METRICS) + PRICE_METRICS
    category_weights = ETF_CATEGORY_WEIGHTS if is_fund else STOCK_CATEGORY_WEIGHTS
    score, rows, category_scores = score_metrics(metrics, values, category_weights)
    return {
        "ticker": ticker.upper(), "name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "is_fund": is_fund, "info": info, "hist": hist, "rows": rows,
        "category_scores": category_scores, "score": score,
    }


def verdict(score):
    if score is None:
        return "Not enough data", "gray"
    if score >= 70:
        return "Strong screen result", "green"
    if score >= 45:
        return "Mixed screen result", "orange"
    return "Weak screen result", "red"


def summary_text(a: dict) -> str:
    """Plain-text snapshot handed to the AI as context."""
    info = a["info"]
    lines = [
        f"{a['name']} ({a['ticker']}) - {'ETF/fund' if a['is_fund'] else 'stock'}",
        f"Sector: {info.get('sector', 'n/a')} | Industry: {info.get('industry', 'n/a')}",
        f"Price: {info.get('currentPrice') or info.get('regularMarketPrice', 'n/a')} {info.get('currency', '')}",
        f"Overall rule-based screen score: {a['score']}/100 ({verdict(a['score'])[0]})",
        "Category breakdown:",
    ]
    lines += [
        f"- {r['Category']}: {r['Score']} with {r['Weight']} weight; {r['Contribution']} ({r['Coverage']})"
        for r in a["category_scores"]
    ]
    lines.append("Measures:")
    lines += [f"- {r['Measure']}: {r['Value']} -> {r['Verdict']}. {r['Why']}" for r in a["rows"]]
    return "\n".join(lines)
