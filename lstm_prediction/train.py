# -*- coding: utf-8 -*-
"""
Training LSTM model for stock price prediction using yfinance data.
The model predicts future stock prices without using a database.
"""
"""
coditor testing
"""

import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from utils import create_dataset, normalize_data

def train_model(ticker, look_back=20, horizon=30, epochs=50, batch_size=8):
    # 1) Fetch data from yfinance
    data = yf.download(ticker, period="1y", interval="1d")
    data = data[['Close']]

    if data.empty:
        raise ValueError(f"No data found for {ticker}")

    # 2) Normalize data
    scaled, scaler = normalize_data(data)
    
    # 3) Create dataset for training (X = features, y = labels)
    X, y = create_dataset(scaled, look_back, horizon)
    X = X.reshape(-1, look_back, 1)  # LSTM needs 3D input
    
    # 4) Build the LSTM model
    model = Sequential([
        LSTM(64, input_shape=(look_back, 1)),
        Dense(horizon)  # Output horizon number of neurons
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # 5) Train the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    # Save the model and scaler
    model.save(f"{ticker}_lstm_model.keras") 

    # Return the trained model and scaler for prediction
    return model, scaler


if __name__ == "__main__":
    # Example usage: Train model for a given stock ticker
    ticker = 'AAPL'  # Apple stock
    model, scaler = train_model(ticker)
    print(f"Model trained and saved for {ticker}")
