import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.set_page_config(page_title="Stock Forecast", layout="wide")

st.title("📈 Stock Price Forecast using ARIMA")

ticker = st.text_input(
    "Enter NSE/BSE Ticker",
    value="RELIANCE.NS"
)

if st.button("Run Forecast"):

    try:
        end_date = datetime.today()
        start_date = end_date - pd.DateOffset(years=5)

        with st.spinner("Downloading stock data..."):

            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )

        if df.empty:
            st.error("No data available.")
            st.stop()

        # Convert to Series
        close_prices = df["Close"].squeeze()

        # Remove missing values
        close_prices = close_prices.dropna()

        st.subheader("Last 5 Years Stock Price")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=close_prices.index,
                y=close_prices.values,
                mode="lines",
                name="Close Price"
            )
        )

        fig.update_layout(
            title=f"{ticker} Historical Closing Prices",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Latest Price", f"{close_prices.iloc[-1]:.2f}")
        col2.metric("Highest", f"{close_prices.max():.2f}")
        col3.metric("Lowest", f"{close_prices.min():.2f}")
        col4.metric("Data Points", len(close_prices))

        st.subheader("ARIMA Forecast")

        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()

        target_date = pd.Timestamp("2027-06-30")

        days_to_forecast = (
            target_date - close_prices.index[-1]
        ).days

        forecast = model_fit.forecast(steps=days_to_forecast)

        predicted_price = forecast.iloc[-1]

        st.success(
            f"Predicted {ticker} Price on June 30, 2027: ₹{predicted_price:.2f}"
        )

        future_dates = pd.date_range(
            start=close_prices.index[-1] + pd.Timedelta(days=1),
            periods=days_to_forecast,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast.values
        })

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=close_prices.index,
                y=close_prices.values,
                mode="lines",
                name="Historical"
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast"],
                mode="lines",
                name="Forecast"
            )
        )

        fig2.update_layout(
            title=f"{ticker} Forecast till June 2027",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=650
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Recent Data")

        st.dataframe(
            df.tail(10),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")
