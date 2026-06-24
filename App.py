import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

st.set_page_config(
    page_title="Stock Forecast using ARIMA",
    layout="wide"
)

st.title("Stock Price Forecasting using ARIMA")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

if st.button("Run Forecast"):

    with st.spinner("Downloading Data..."):

        stock = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=True
        )

    if stock.empty:
        st.error("Invalid ticker symbol")
        st.stop()

    close_prices = stock["Close"]

    st.subheader("Historical Stock Prices (5 Years)")

    fig = px.line(
        stock,
        x=stock.index,
        y="Close",
        title=f"{ticker} Closing Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ARIMA Forecast")

    try:

        model = ARIMA(close_prices, order=(5,1,0))
        model_fit = model.fit()

        forecast_days = 365

        forecast = model_fit.forecast(
            steps=forecast_days
        )

        future_dates = pd.date_range(
            start=stock.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast.values
        })

        june_2027 = forecast_df[
            (forecast_df["Date"].dt.year == 2027) &
            (forecast_df["Date"].dt.month == 6)
        ]

        if not june_2027.empty:

            june_forecast = june_2027.iloc[-1]["Forecast"]

            st.success(
                f"Predicted Price for June 2027: "
                f"${june_forecast:.2f}"
            )

        historical_df = pd.DataFrame({
            "Date": stock.index,
            "Price": close_prices
        })

        future_plot_df = pd.DataFrame({
            "Date": forecast_df["Date"],
            "Price": forecast_df["Forecast"]
        })

        historical_df["Type"] = "Historical"
        future_plot_df["Type"] = "Forecast"

        combined = pd.concat(
            [historical_df, future_plot_df]
        )

        fig2 = px.line(
            combined,
            x="Date",
            y="Price",
            color="Type",
            title=f"{ticker} ARIMA Forecast"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.subheader("June 2027 Forecast Data")
        st.dataframe(june_2027)

    except Exception as e:
        st.error(f"Error: {e}")
