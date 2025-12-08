"""
Prediction Script
Predicts stock prices for the next 7 days and provides buy/sell indicator
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime, timedelta
import yfinance as yf
import json
import os
import sys


def get_recent_data_and_sentiment(ticker, look_back=100):
    """Get recent stock data and calculate current sentiment"""

    # Fetch recent stock data
    # Use a longer calendar window to make sure we have >= look_back TRADING days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=look_back * 3)  # e.g. 300 calendar days

    print(
        f"Fetching recent stock data from "
        f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}..."
    )

    stock_data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False,)
    stock_data.reset_index(inplace=True)

    if stock_data.empty:
        print(f"Warning: No recent stock data found for {ticker}")
        return None, None

    if len(stock_data) < look_back:
        print(
            f"Error: Not enough recent trading days for look_back={look_back}. "
            f"Got only {len(stock_data)} rows."
        )
        return None, None

    # Get recent news and calculate sentiment
    avg_sentiment = [0.2, 0.2, 0.2, 0.2, 0.2]

    sentiment_path = f"data/news/{ticker}_daily_scores.json"
    if os.path.exists(sentiment_path):
        try:
            with open(sentiment_path, "r", encoding="utf-8") as f:
                sentiment_data = json.load(f)

            if isinstance(sentiment_data, dict) and sentiment_data:
                # Get most recent sentiment window (up to 7 days)
                recent_dates = sorted(sentiment_data.keys())[-7:]
                recent_sentiments = [
                    np.array(sentiment_data[d], dtype=float)
                    for d in recent_dates
                    if d in sentiment_data
                ]

                if recent_sentiments:
                    avg = np.mean(recent_sentiments, axis=0)
                    # If it's basically all 0.2 anyway, treat as neutral
                    if not np.allclose(avg, np.full_like(avg, 0.2), atol=1e-3):
                        avg_sentiment = avg.tolist()
        except Exception as e:
            print(f"Warning: Could not load sentiment data: {e}")

    return stock_data, avg_sentiment


def predict_next_7_days(ticker, look_back=100):
    """Predict stock prices for the next 7 days"""

    # Check if model exists
    model_path = f"models/{ticker}_lstm_model.h5"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using 4_model_training.py")
        return None

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    scaler_features_path = f"models/{ticker}_scaler_features.pkl"
    scaler_target_path = f"models/{ticker}_scaler_target.pkl"
    feature_cols_path = f"models/{ticker}_feature_cols.pkl"

    if not all(
        os.path.exists(p)
        for p in [scaler_features_path, scaler_target_path, feature_cols_path]
    ):
        print("Error: Scaler files not found. Please retrain the model.")
        return None

    with open(scaler_features_path, "rb") as f:
        scaler_features = pickle.load(f)

    with open(scaler_target_path, "rb") as f:
        scaler_target = pickle.load(f)

    with open(feature_cols_path, "rb") as f:
        feature_cols = pickle.load(f)

    # Get recent data
    stock_data, current_sentiment = get_recent_data_and_sentiment(
        ticker, look_back=look_back
    )

    if stock_data is None:
        return None

    # Prepare features
    if "Adj Close" in stock_data.columns:
        stock_data = stock_data.drop(columns=["Adj Close"])

    # Add sentiment columns (same current_sentiment for all recent rows)
    for i, sent in enumerate(current_sentiment, 1):
        stock_data[f"sentiment_{i}"] = float(sent)

    # Ensure all feature columns exist
    for col in feature_cols:

        if col not in stock_data.columns:

            if "sentiment" in col:
                stock_data[col] = 0.2
            else:
                # Try to find a similar column or use mean
                if "open" in col.lower() and "Open" in stock_data.columns:
                    stock_data[col] = stock_data["Open"]
                elif "high" in col.lower() and "High" in stock_data.columns:
                    stock_data[col] = stock_data["High"]
                elif "low" in col.lower() and "Low" in stock_data.columns:
                    stock_data[col] = stock_data["Low"]
                elif "volume" in col.lower() and "Volume" in stock_data.columns:
                    stock_data[col] = stock_data["Volume"]
                else:
                    stock_data[col] = stock_data["Close"].mean()

    # Work in unscaled feature space for the rolling window
    feature_matrix = stock_data[feature_cols].values

    if len(feature_matrix) < look_back:
        print(
            f"Error: Only {len(feature_matrix)} rows of features available, "
            f"need at least {look_back}."
        )
        return None

    # Last look_back unscaled rows
    recent_unscaled = feature_matrix[-look_back:, :].copy()
    n_features = recent_unscaled.shape[1]

    # Precompute indices of price-related columns we want to scale with price
    price_feature_indices = [
        i
        for i, col in enumerate(feature_cols)
        if any(x in col.lower() for x in ["open", "high", "low", "close"])
    ]
    if not price_feature_indices:
        print("Warning: No price-related features found; using static features for multi-step prediction.")

    # Last known actual close price
    last_actual_close = float(stock_data["Close"].iloc[-1])

    # Predict next 7 days
    print("Predicting next 7 days...")
    predictions = []

    for day in range(7):
        # Scale the current unscaled window
        recent_scaled = scaler_features.transform(recent_unscaled)
        current_input = recent_scaled.reshape(1, look_back, n_features)

        # 1 step-ahead prediction
        pred_scaled = model.predict(current_input, verbose=0)
        pred_price = scaler_target.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred_price)

        # Build next day's unscaled feature row
        new_row_unscaled = recent_unscaled[-1].copy()

        if day == 0:
            base_close = last_actual_close
        else:
            base_close = predictions[-2]

        price_change_ratio = pred_price / base_close if base_close > 0 else 1.0

        # Update all price-related columns in unscaled space
        for idx in price_feature_indices:
            new_row_unscaled[idx] *= price_change_ratio

        # Roll the window: drop first, append new row
        recent_unscaled = np.vstack([recent_unscaled[1:], new_row_unscaled])

    return predictions



def generate_buy_sell_indicator(current_price, predictions):
    """Generate buy/sell indicator based on predictions"""
    avg_predicted_price = np.mean(predictions)
    price_change_pct = ((avg_predicted_price - current_price) / current_price) * 100

    # Calculate trend (are predictions increasing or decreasing?)
    if len(predictions) > 1:
        trend = (predictions[-1] - predictions[0]) / predictions[0] * 100
    else:
        trend = price_change_pct

    # Strategy:
    # - Buy if predicted average is > 2% higher AND trend is positive
    # - Sell if predicted average is < -2% lower AND trend is negative
    # - Hold otherwise

    if price_change_pct > 2 and trend > 0:
        indicator = "BUY"
        confidence = min(abs(price_change_pct) / 10, 0.9)  # cap at 90%
    elif price_change_pct < -2 and trend < 0:
        indicator = "SELL"
        confidence = min(abs(price_change_pct) / 10, 0.9)
    elif price_change_pct > 1:
        indicator = "WEAK BUY"
        confidence = min(abs(price_change_pct) / 5, 0.7)
    elif price_change_pct < -1:
        indicator = "WEAK SELL"
        confidence = min(abs(price_change_pct) / 5, 0.7)
    else:
        indicator = "HOLD"
        # When it's basically flat, don't pretend to be super confident
        confidence = max(0.4, 1.0 - (abs(price_change_pct) / 5))

    return indicator, confidence, price_change_pct, trend


def main_prediction(ticker):
    """Main prediction function"""

    print(f"\n{'='*70}")
    print(f"Stock Prediction for {ticker}")
    print(f"{'='*70}\n")

    # Get current price
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")

        if hist.empty:
            print(f"Error: Could not fetch current price for {ticker}")
            return None
        current_price = hist["Close"].iloc[-1]
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return None

    print(f"Current Price: ${current_price:.2f}\n")

    #Predict next 7 days
    predictions = predict_next_7_days(ticker)

    if predictions is None:
        return None

    # Generate dates
    dates = [
        (datetime.now() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(7)
    ]

    # Display predictions
    print("\n" + "=" * 70)
    print("7-Day Price Predictions:")
    print("=" * 70)
    print(f"{'Date':<12} {'Predicted Price':<18} {'Change %':<12} {'Change $':<12}")
    print("-" * 70)

    for date, pred in zip(dates, predictions):
        change_pct = ((pred - current_price) / current_price) * 100
        change_dollar = pred - current_price
        print(
            f"{date:<12} ${pred:>10.2f}      {change_pct:>+8.2f}%    ${change_dollar:>+8.2f}"
        )

    # Generate buy/sell indicator
    indicator, confidence, price_change, trend = generate_buy_sell_indicator(
        current_price, predictions
    )

    print("\n" + "=" * 70)
    print("Trading Recommendation:")
    print("=" * 70)
    print(f"Recommendation: {indicator}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"Expected 7-day average change: {price_change:+.2f}%")
    print(f"Price trend (Day 1 to Day 7): {trend:+.2f}%")
    print(f"Average predicted price: ${np.mean(predictions):.2f}")
    print(
        f"Predicted price range: ${min(predictions):.2f} - ${max(predictions):.2f}"
    )
    print("=" * 70 + "\n")

    return {
        "ticker": ticker,
        "current_price": current_price,
        "predictions": dict(zip(dates, predictions)),
        "indicator": indicator,
        "confidence": confidence,
        "expected_change_pct": price_change,
        "trend_pct": trend,
    }


if __name__ == "__main__":
    # Default ticker
    TICKER = "AAPL"
    if len(sys.argv) > 1:
        TICKER = sys.argv[1].upper()

    result = main_prediction(TICKER)

    if result:
        print("\nPrediction completed successfully!")
    else:
        print("\nPrediction failed. Please check the error messages above.")
