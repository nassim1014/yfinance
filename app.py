import pandas as pd
import streamlit as st

from core import POPULAR_ETFS, POPULAR_STOCKS, analyze, fmt, summary_text, verdict

st.set_page_config(page_title="Stock Check", page_icon="📈", layout="wide")
st.title("📈 Stock Check")
st.caption("Live data from Yahoo Finance · educational tool, not financial advice")

with st.sidebar:
    st.header("Pick an asset")
    kind = st.radio("Type", ["Stocks", "ETFs"], horizontal=True)
    options = POPULAR_STOCKS if kind == "Stocks" else POPULAR_ETFS
    choice = st.selectbox("Popular", list(options))
    custom = st.text_input("…or type any ticker", placeholder="e.g. AMD, MC.PA")
    ticker = (custom.strip() or options[choice]).upper()

try:
    with st.spinner(f"Fetching {ticker}…"):
        a = analyze(ticker)
except Exception as e:
    st.error(str(e))
    st.stop()

# remember for the AI page
st.session_state["analysis"] = {"ticker": a["ticker"], "name": a["name"], "summary": summary_text(a)}

info, hist = a["info"], a["hist"]
label, color = verdict(a["score"])

st.subheader(f"{a['name']} ({a['ticker']})")
c1, c2, c3, c4 = st.columns(4)
price = info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1]
c1.metric("Price", f"{price:,.2f} {info.get('currency', '')}")
c2.metric("Market cap" if not a["is_fund"] else "Fund size", fmt(info.get("marketCap") or info.get("totalAssets"), "big"))
c3.metric("Sector" if not a["is_fund"] else "Category", info.get("sector") or info.get("category") or "n/a")
c4.metric("Screen score", f"{a['score']}/100" if a["score"] is not None else "n/a")
st.markdown(f"### :{color}[{label}]")

period = st.select_slider("Chart period", ["1M", "6M", "1Y", "3Y", "5Y"], value="1Y")
days = {"1M": 21, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260}[period]
st.line_chart(hist["Close"].tail(days), height=280)

st.subheader("How the score is calculated")
st.dataframe(pd.DataFrame(a["category_scores"]), hide_index=True, use_container_width=True)
st.caption(
    "Each measure is rated Good (100), Fair (50), or Weak (0). Measures are averaged within "
    "each category, then the category scores are combined using the displayed weights. A category "
    "with no available data is excluded and the remaining weights are rescaled."
)

st.subheader("Key measures")
st.dataframe(pd.DataFrame(a["rows"]), hide_index=True, use_container_width=True)
st.caption(
    "The score is a rule-based screen, not an investment recommendation. Thresholds are generic "
    "rules of thumb: always compare companies with their sector peers. "
    "See the **Glossary** page for what each measure means, and **Ask AI** to chat about this asset."
)
