import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

st.set_page_config(
    page_title="AI Equity Research Platform",
    layout="wide"
)

st.title("AI Equity Research Platform")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    "AAPL"
).upper()


@st.cache_data(ttl=3600)
def load_data(symbol):

    try:

        data = yf.download(
            symbol,
            period="5y",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or len(data) == 0:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    except Exception:
        return pd.DataFrame()


data = load_data(ticker)

if data.empty:
    st.error(
        f"No data found for '{ticker}'. Try another ticker such as AAPL, MSFT, TSLA, INFY.NS or RELIANCE.NS"
    )
    st.stop()

if "Close" not in data.columns:
    st.error("Close price column missing.")
    st.stop()

close = data["Close"].squeeze()

# ==========================================
# Technical Indicators
# ==========================================

data["SMA20"] = SMAIndicator(
    close=close,
    window=20
).sma_indicator()

data["SMA50"] = SMAIndicator(
    close=close,
    window=50
).sma_indicator()

data["SMA200"] = SMAIndicator(
    close=close,
    window=200
).sma_indicator()

data["EMA20"] = EMAIndicator(
    close=close,
    window=20
).ema_indicator()

data["EMA50"] = EMAIndicator(
    close=close,
    window=50
).ema_indicator()

data["RSI"] = RSIIndicator(
    close=close,
    window=14
).rsi()

macd = MACD(close)

data["MACD"] = macd.macd()
data["MACD_SIGNAL"] = macd.macd_signal()
data["MACD_HIST"] = macd.macd_diff()

bb = BollingerBands(
    close=close,
    window=20,
    window_dev=2
)

data["BB_UPPER"] = bb.bollinger_hband()
data["BB_LOWER"] = bb.bollinger_lband()

# ==========================================
# Sidebar Snapshot
# ==========================================

st.sidebar.header("Current Snapshot")

st.sidebar.metric(
    "Current Price",
    f"{float(close.iloc[-1]):.2f}"
)

high_52 = float(close.tail(252).max())
low_52 = float(close.tail(252).min())

st.sidebar.metric(
    "52 Week High",
    f"{high_52:.2f}"
)

st.sidebar.metric(
    "52 Week Low",
    f"{low_52:.2f}"
)

# ==========================================
# Tabs
# ==========================================

tab1, tab2, tab3 = st.tabs(
    [
        "Price Chart",
        "Technical Analysis",
        "Fundamental Analysis"
    ]
)

# ==========================================
# PRICE TAB
# ==========================================

with tab1:

    st.subheader(f"{ticker} Price Analysis")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=close,
            mode="lines",
            name="Close"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA20"],
            mode="lines",
            name="SMA 20"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA50"],
            mode="lines",
            name="SMA 50"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA200"],
            mode="lines",
            name="SMA 200"
        )
    )

    fig.update_layout(
        height=650,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# TECHNICAL TAB
# ==========================================

with tab2:

    st.subheader("Relative Strength Index (RSI)")

    st.line_chart(
        data["RSI"]
    )

    latest_rsi = float(
        data["RSI"].dropna().iloc[-1]
    )

    st.metric(
        "Current RSI",
        round(latest_rsi, 2)
    )

    if latest_rsi > 70:
        st.warning("Overbought Zone")
    elif latest_rsi < 30:
        st.success("Oversold Zone")
    else:
        st.info("Neutral Zone")

    st.divider()

    st.subheader("MACD")

    macd_df = pd.DataFrame({
        "MACD": data["MACD"],
        "Signal": data["MACD_SIGNAL"]
    })

    st.line_chart(macd_df)

    st.divider()

    st.subheader("Bollinger Bands")

    bb_df = pd.DataFrame({
        "Close": close,
        "Upper Band": data["BB_UPPER"],
        "Lower Band": data["BB_LOWER"]
    })

    st.line_chart(bb_df)

# ==========================================
# FUNDAMENTALS TAB
# ==========================================

with tab3:

    st.subheader("Fundamental Metrics")

    try:

        stock = yf.Ticker(ticker)
        info = stock.info

        fundamentals = {
            "Market Cap": info.get("marketCap"),
            "Trailing PE": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "PEG Ratio": info.get("pegRatio"),
            "Price To Book": info.get("priceToBook"),
            "Dividend Yield": info.get("dividendYield"),
            "ROE": info.get("returnOnEquity"),
            "Revenue Growth": info.get("revenueGrowth"),
            "Profit Margin": info.get("profitMargins"),
            "Debt To Equity": info.get("debtToEquity")
        }

        fundamentals_df = pd.DataFrame(
            fundamentals.items(),
            columns=["Metric", "Value"]
        )

        st.dataframe(
            fundamentals_df,
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Fundamental data currently unavailable: {e}"
        )
