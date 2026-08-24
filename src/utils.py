"""Shared helper functions used across the project."""

import pickle
import numpy as np
import pandas as pd

def load_data(dataset_path:str):
    data = pd.read_csv(dataset_path)
    data.index = pd.to_datetime(data["Date"])
    return data.sort_index()

def count_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

def check_ohlc_validity(data):
    invalid_high = (data["High"] < data["Open"]) | (data["High"] < data["Close"]) | (data["High"] < data["Low"])
    invalid_low = (data["Low"] > data["Open"]) | (data["Low"] > data["Close"]) | (data["Low"] > data["High"])
    invalid_rows = invalid_high | invalid_low
    print("invalid OHLC rows:", invalid_rows.sum())
    return invalid_rows

def check_negative_prices(data):
    negative_rows = (data["Open"] <= 0) | (data["Close"] <= 0) | (data["High"] <= 0) | (data["Low"] <= 0)
    print("negative or zero price rows:", negative_rows.sum())
    return negative_rows

def calculate_volatility(data, window=20):
    data = data.copy()
    data["daily_return"] = data["Close"].pct_change()
    data["Volatility"] = data["daily_return"].rolling(window=window).std()
    return data

def calculate_bullish_bearish(data):
    data = data.copy()
    data["candle_type"] = np.where(data["Close"] > data["Open"], "bullish",
                            np.where(data["Close"] < data["Open"], "bearish", "neutral"))
    print(data["candle_type"].value_counts())
    return data

def load_model_data(file_path):
    """
    Load all dataset splits from a pickle file and return them.
    """
    with open(file_path, 'rb') as f:
        all_data = pickle.load(f)
        
    print("Scaler & All data successfully loaded and ready for the model!")
    
    return (
        all_data['X_train'], all_data['y_train'],
        all_data['X_valid'], all_data['y_valid'],
        all_data['X_test'], all_data['y_test']
    )