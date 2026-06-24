```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

import plotly.graph_objects as go

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from statsmodels.tsa.arima.model import ARIMA

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

    data = yf.download(
        symbol,
        period="5y",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


data = load_data(ticker)

if data.empty:
    st.error("No data found.")
    st.stop()

close = data["Close"].squeeze()

# =====================================
# Technical Indicators
# =====================================

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

bb = BollingerBands(
    close=close,
    window=20,
    window_dev=2
)

data["BB_UPPER"] = bb.bollinger_hband()
data["BB_LOWER"] = bb.bollinger_lband()

# =====================================
# Sidebar Metrics
# =====================================

st.sidebar.header("Market Snapshot")

current_price = float(close.iloc[-1])

st.sidebar.metric(
    "Current Price",
    f"{current_price:.2f}"
)

st.sidebar.metric(
    "52W High",
    f"{close.tail(252).max():.2f}"
)

st.sidebar.metric(
    "52W Low",
    f"{close.tail(252).min():.2f}"
)

# =====================================
# Tabs
# =====================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Price Analysis",
        "Technical Indicators",
        "Risk Analytics",
        "Forecasting"
    ]
)

# =====================================
# PRICE ANALYSIS
# =====================================

with tab1:

    st.subheader(f"{ticker} Price Analysis")

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
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# TECHNICAL ANALYSIS
# =====================================

with tab2:

    st.subheader("RSI")

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
        st.warning("Overbought")

    elif latest_rsi < 30:
        st.success("Oversold")

    else:
        st.info("Neutral")

    st.divider()

    st.subheader("MACD")

    st.line_chart(
        pd.DataFrame({
            "MACD": data["MACD"],
            "Signal": data["MACD_SIGNAL"]
        })
    )

    st.divider()

    st.subheader("Bollinger Bands")

    st.line_chart(
        pd.DataFrame({
            "Close": close,
            "Upper": data["BB_UPPER"],
            "Lower": data["BB_LOWER"]
        })
    )

# =====================================
# RISK ANALYTICS
# =====================================

with tab3:

    returns = close.pct_change().dropna()

    annual_vol = (
        returns.std() * np.sqrt(252)
    ) * 100

    sharpe = (
        returns.mean() /
        returns.std()
    ) * np.sqrt(252)

    drawdown = (
        close /
        close.cummax()
    ) - 1

    max_drawdown = (
        drawdown.min()
    ) * 100

    total_return = (
        (
            close.iloc[-1]
            /
            close.iloc[0]
        ) - 1
    ) * 100

    risk_df = pd.DataFrame({
        "Metric": [
            "5 Year Return %",
            "Annual Volatility %",
            "Sharpe Ratio",
            "Max Drawdown %"
        ],
        "Value": [
            round(total_return, 2),
            round(annual_vol, 2),
            round(sharpe, 2),
            round(max_drawdown, 2)
        ]
    })

    st.dataframe(
        risk_df,
        use_container_width=True
    )

# =====================================
# FORECASTING
# =====================================

with tab4:

    st.subheader("ARIMA Forecast")

    monthly = close.resample("ME").last()

    model = ARIMA(
        monthly,
        order=(2,1,2)
    )

    fitted = model.fit()

    forecast = fitted.forecast(
        steps=12
    )

    future_dates = pd.date_range(
        start=monthly.index[-1] +
        pd.offsets.MonthEnd(1),
        periods=12,
        freq="ME"
    )

    forecast_fig = go.Figure()

    forecast_fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly,
            name="Historical"
        )
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast,
            name="Forecast"
        )
    )

    forecast_fig.update_layout(
        height=600,
        template="plotly_white"
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )

    st.dataframe(
        pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast
        })
    )
```
