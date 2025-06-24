import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_data(df):
    """
    Normalizes the data using MinMaxScaler.
    Args:
        df (pandas.DataFrame): DataFrame containing stock prices.
    Returns:
        scaled (numpy.ndarray): Normalized data.
        scaler (MinMaxScaler): Scaler used to normalize the data.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])  # Normalize only 'Close' prices
    return scaled, scaler


def create_dataset(scaled, look_back=20, horizon=30):
    """
    Converts the time-series data into a supervised learning format.
    Args:
        scaled (numpy.ndarray): Normalized stock prices.
        look_back (int): The number of previous days to use as input for prediction.
        horizon (int): The number of days to predict ahead.
    Returns:
        X (numpy.ndarray): Features (inputs) for training.
        y (numpy.ndarray): Labels (outputs) for training.
    """
    X, y = [], []
    for i in range(look_back, len(scaled) - horizon + 1):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i:i + horizon, 0])
    return np.array(X), np.array(y)
