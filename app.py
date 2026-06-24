import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="Stock Price Forecast using ARIMA",
    layout="wide"
)

st.title("Stock Price Forecasting using ARIMA")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

if st.button("Run Forecast"):

    try:
        stock = yf.download(
            ticker,
            period="5y",
            auto_adjust=True,
            progress=False
        )

        if stock.empty:
            st.error("No data found for this ticker.")
            st.stop()

        # Fix for yfinance column structure
        close_prices = stock["Close"]

        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        close_prices = close_prices.dropna()

        st.subheader("Historical Stock Prices (5 Years)")

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
            title=f"{ticker} Closing Price",
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("ARIMA Forecast")

        model = ARIMA(
            close_prices,
            order=(5, 1, 0)
        )

        model_fit = model.fit()

        forecast_days = 365

        forecast = model_fit.forecast(
            steps=forecast_days
        )

        future_dates = pd.date_range(
            start=close_prices.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast
        })

        june_2027 = forecast_df[
            (forecast_df["Date"].dt.year == 2027)
            &
            (forecast_df["Date"].dt.month == 6)
        ]

        if not june_2027.empty:

            june_price = float(
                june_2027.iloc[-1]["Forecast"]
            )

            st.success(
                f"Predicted Stock Price for June 2027: ${june_price:.2f}"
            )

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=close_prices.index,
                y=close_prices.values,
                mode="lines",
                name="Historical"
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast"],
                mode="lines",
                name="Forecast"
            )
        )

        forecast_fig.update_layout(
            title=f"{ticker} Historical vs Forecast",
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        st.subheader("June 2027 Forecast Values")
        st.dataframe(june_2027)

    except Exception as e:
        st.error(f"Error: {str(e)}")
