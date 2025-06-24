# -*- coding: utf-8 -*-
"""
Using trained LSTM model to make predictions on historical stock prices.
"""
import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import create_dataset, normalize_data

def predict_with_model(ticker, look_back=20, horizon=30):
    # 1) Load the trained model
    model = tf.keras.models.load_model(f"{ticker}_lstm_model.keras")

    # 2) Fetch the latest data for prediction
    data = yf.download(ticker, period="1y", interval="1d")
    data = data[['Close']]

    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # 3) Normalize the data
    scaled, scaler = normalize_data(data)
    
    # 4) Prepare dataset for prediction
    X, y = create_dataset(scaled, look_back, horizon)
    X = X.reshape(-1, look_back, 1)

    # 5) Make predictions
    predictions = model.predict(X, verbose=0)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)

    # 6) Get real values (actual closing prices) for comparison
    real_1 = scaler.inverse_transform(y[:, 0:1]).ravel()  # Get the real closing prices

    # 7) Calculate error metrics
    mse = mean_squared_error(real_1, predictions[:, 0])  # MSE for the first step prediction
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_1, predictions[:, 0])
    r2 = r2_score(real_1, predictions[:, 0])

    # 8) Plot results
    plt.figure(figsize=(14, 6))
    
    # Ensure prediction_dates and real_1 have the same length
    prediction_dates = data.index[look_back:]  # Dates for prediction data
    
    # Trim prediction_dates to match real_1 length
    if len(prediction_dates) != len(real_1):
        prediction_dates = prediction_dates[:len(real_1)]
    
    plt.plot(prediction_dates, real_1, label="Actual Prices", color="blue")
    plt.plot(prediction_dates, predictions[:, 0], label="Predicted Prices", color="orange")
    
    plt.title(f"{ticker} Stock Price Prediction vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()

    # Display metrics on the graph
    plt.text(0.02, 0.98,
             f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}",
             transform=plt.gca().transAxes, va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Show the plot
    plt.show()

    # Return predictions and metrics
    return predictions, {"mse": mse, "rmse": rmse, "mae": mae, "r2_score": r2}


if __name__ == "__main__":
    # Example usage: Predict stock prices for Apple
    ticker = 'AAPL'  # Apple stock
    predictions, metrics = predict_with_model(ticker)
    print(f"Predictions for {ticker}:")
    print(predictions[-5:])  # Print the last 5 predictions
    print(f"Metrics: {metrics}")
