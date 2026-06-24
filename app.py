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


@st.cache_data
def load_data(ticker):
    data = yf.download(
        ticker,
        period="5y",
        auto_adjust=True,
        progress=False
    )

    # Fix MultiIndex issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


data = load_data(ticker)

if data.empty:
    st.error("No data found.")
    st.stop()

close = data["Close"]

# ==========================
# Technical Indicators
# ==========================

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

# ==========================
# Tabs
# ==========================

tab1, tab2, tab3 = st.tabs(
    [
        "Price Chart",
        "Technical Analysis",
        "Fundamentals"
    ]
)

# ==========================
# PRICE TAB
# ==========================

with tab1:

    st.subheader(f"{ticker} Price Chart")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=close,
            name="Close"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA20"],
            name="SMA20"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA50"],
            name="SMA50"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA200"],
            name="SMA200"
        )
    )

    fig.update_layout(
        height=600,
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================
# TECHNICAL TAB
# ==========================

with tab2:

    st.subheader("RSI")

    rsi_df = pd.DataFrame({
        "RSI": data["RSI"]
    })

    st.line_chart(rsi_df)

    current_rsi = float(
        data["RSI"].dropna().iloc[-1]
    )

    st.metric(
        "Current RSI",
        round(current_rsi, 2)
    )

    if current_rsi > 70:
        st.warning("Overbought")
    elif current_rsi < 30:
        st.success("Oversold")
    else:
        st.info("Neutral")

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

# ==========================
# FUNDAMENTALS TAB
# ==========================

with tab3:

    st.subheader("Fundamental Analysis")

    try:

        info = yf.Ticker(ticker).info

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

        st.error(
            f"Fundamental data unavailable: {e}"
        )

# ==========================
# SIDEBAR
# ==========================

st.sidebar.header("Current Snapshot")

st.sidebar.metric(
    "Current Price",
    round(float(close.iloc[-1]), 2)
)

st.sidebar.metric(
    "52 Week High",
    round(float(close.tail(252).max()), 2)
)

st.sidebar.metric(
    "52 Week Low",
    round(float(close.tail(252).min()), 2)
)
