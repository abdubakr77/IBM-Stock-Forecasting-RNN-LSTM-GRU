"""Load raw IBM stock data, clean it, scale it, and create train/valid/test sequences."""
from src.utils import check_negative_prices, check_ohlc_validity, calculate_volatility
import pandas as pd


def load_data(dataset_path:str):
    df = pd.read_csv(dataset_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)


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


