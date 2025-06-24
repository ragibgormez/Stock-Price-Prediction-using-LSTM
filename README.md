# Stock-Price-Prediction-using-LSTM

This project demonstrates how to use an LSTM (Long Short-Term Memory) model to predict stock prices based on historical data. The model is trained using yfinance to fetch real-time stock data and TensorFlow/Keras to build and train the LSTM model.

## Table of Contents:
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How to Use](#how-to-use)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#making-predictions)
- [Code Explanation](#code-explanation)
  - [train.py](#trainpy)
  - [predict.py](#predictpy)
  - [utils.py](#utilspy)
- [Parameter Explanation](#parameter-explanation)
- [Metrics](#metrics)
- [Conclusion](#conclusion)


## Project Overview
This project uses an LSTM (Long Short-Term Memory) model to predict future stock prices based on historical stock data. It pulls stock data via the yfinance API and processes it using TensorFlow and Keras to build the predictive model. The project demonstrates how to train, save, and load an LSTM model for making predictions on financial data.

## Technologies Used
- **Python 3.x**: Programming language used for the entire project.
- **yfinance**: Library to fetch historical stock data.
- **TensorFlow / Keras**: Framework used for building and training the LSTM model.
- **scikit-learn**: For preprocessing the data and splitting it into training/testing sets.
- **matplotlib**: For plotting and visualizing the actual vs predicted stock prices.
- **joblib**: For saving and loading the trained models.
- **pandas**: For data manipulation and analysis.

## Installation

1.  **Clone the repository:**

        git clone https://github.com/ragibgormez/stock-price-prediction-lstm.git
        cd stock-price-prediction-lstm

2.  **Create a virtual environment:**

        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3.  **Install the dependencies:**

        pip install -r requirements.txt

    The `requirements.txt` file should contain:

        yfinance
        numpy
        pandas
        tensorflow
        scikit-learn
        matplotlib
        joblib

## How to Use

### Training the Model
To train the LSTM model, run the following command:

    python train.py

**What happens when you run `train.py`:**
- The script fetches 1 year of historical stock data for a specific ticker symbol using `yfinance`.
- It preprocesses the data, normalizes it, and prepares it for training.
- It trains an LSTM model on the data.
- After training, the model is saved in the `.keras` format for future use.

### Making Predictions
After training the model, you can use the `predict.py` script to make predictions.

    python predict.py

**What happens when you run `predict.py`:**
- The script loads the trained LSTM model from the `.keras` file.
- It fetches the latest stock data for the specified ticker symbol.
- It makes predictions on the next days of stock prices.
- It plots the actual vs predicted stock prices and calculates the following error metrics: MSE, RMSE, MAE, and R².

## Code Explanation

### train.py
This script is responsible for training the LSTM model:
- `train_model()`: Fetches stock data, preprocesses it, and trains the LSTM model.
- **Model Training**: The LSTM model is trained using the historical stock data and saved after training.

### predict.py
This script uses the trained model to make predictions:
- `predict_with_model()`: Loads the saved model, fetches the latest stock data, and makes predictions using the model.
- **Plotting**: The results are plotted, showing actual vs predicted stock prices, and the error metrics are displayed on the plot.

### utils.py
This file contains helper functions used for data preprocessing:
- `normalize_data()`: Normalizes the stock data using the `MinMaxScaler`.
- `create_dataset()`: Prepares the data for the LSTM model by creating X (features) and y (labels).

## Parameter Explanation
- **`look_back`**: Number of previous days the model will consider to predict the next day.
  - *Example*: `look_back=20` means the model will use the last 20 days' data to predict the next day's stock price.
- **`horizon`**: Number of future days the model will predict.
  - *Example*: `horizon=30` means the model will predict the next 30 days' stock prices.
- **`epochs`**: The number of times the model will process the entire training dataset.
  - More epochs generally result in better training but may also cause overfitting if too high.
- **`batch_size`**: The number of training examples utilized in one update of the model’s weights.
  - A smaller batch size can result in better generalization but may take longer to converge.

## Metrics
The following error metrics are calculated to evaluate the model's performance:
- **MSE (Mean Squared Error)**: Measures the average of the squares of the differences between the actual and predicted prices.
- **RMSE (Root Mean Squared Error)**: The square root of MSE, which gives a more interpretable error value.
- **MAE (Mean Absolute Error)**: Measures the average of the absolute differences between the actual and predicted prices.
- **R² Score (Coefficient of Determination)**: Measures how well the model’s predictions match the actual data. A score closer to 1 indicates better predictions.

## Conclusion
This project provides a simple demonstration of using LSTM to predict stock prices based on historical data. By leveraging `yfinance` for data retrieval and TensorFlow for model training, this model can be used to predict future stock prices and analyze past trends.

Feel free to fork the repository, experiment with different parameters, and improve the model. Contributions are always welcome!
