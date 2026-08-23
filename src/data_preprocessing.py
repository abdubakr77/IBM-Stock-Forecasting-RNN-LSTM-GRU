"""Load raw IBM stock data, clean it, scale it, and create train/valid/test sequences."""
from src.utils import check_negative_prices, check_ohlc_validity, calculate_volatility
import pandas as pd


def load_data(dataset_path:str):
    data = pd.read_csv(dataset_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    return data


def remove_negative_prices(data):
    invalid_rows = check_negative_prices(data)
    invalid_indices = invalid_rows[invalid_rows == True].index.values
    if len(invalid_indices):
        print(f"negative/zero price rows found: {len(invalid_indices)}, removing them")
        data.drop(data.iloc[invalid_indices].index.values, inplace=True)
    else:
        print("no negative or zero price rows found")
    return data.reset_index(drop=True)


def remove_invalid_ohlc(data):
    invalid_rows = check_ohlc_validity(data)
    invalid_indices = invalid_rows[invalid_rows == True].index.values
    if len(invalid_indices):
        print(f"invalid OHLC rows found: {len(invalid_indices)}, removing them")
        data.drop(data.iloc[invalid_indices].index.values, inplace=True)
    else:
        print("no invalid OHLC rows found")
    return data.reset_index(drop=True)


def feature_engineering(data):
    data["Daily Return"] = data["Close"].pct_change()
    data["High Low Range"] = data["High"] - data["Low"]
    data["Open Close Range"] = (data["Close"] - data["Open"]).abs()
    data = calculate_volatility(data, 5)
    return data[['Date', 'Open', 'High', 'Low', 'Close', "Daily Return", 'Volume', "Volatility", "Open Close Range", "High Low Range"]]


def remove_return_outliers(data, upper=0.3, lower=-0.2):
    data = data.copy()
    mask = (data["Daily Return"] < upper) & (data["Daily Return"] > lower) | (data["Daily Return"].isna())
    removed = len(data) - mask.sum()
    if removed:
        print(f"return outlier rows found: {removed}, Removed Successfully!")
    else:
        print("no return outlier rows found")
    data = data[mask]
    data = data.drop(columns=["Daily Return"])
    return data.reset_index(drop=True)
