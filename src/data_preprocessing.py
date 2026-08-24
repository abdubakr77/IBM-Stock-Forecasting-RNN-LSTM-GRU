"""Load raw IBM stock data, clean it, scale it, and create train/valid/test sequences."""
from src.utils import check_negative_prices, check_ohlc_validity, calculate_volatility
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

scaler = MinMaxScaler()


def remove_negative_prices(data):
    invalid_rows = check_negative_prices(data)
    invalid_indices = invalid_rows[invalid_rows == True].index.values
    if len(invalid_indices):
        print(f"negative/zero price rows found: {len(invalid_indices)}, removing them")
        data.drop(data.iloc[invalid_indices].index.values, inplace=True)
    else:
        print("no negative or zero price rows found")
    return data.sort_index()


def remove_invalid_ohlc(data):
    invalid_rows = check_ohlc_validity(data)
    invalid_indices = invalid_rows[invalid_rows == True].index.values
    if len(invalid_indices):
        print(f"invalid OHLC rows found: {len(invalid_indices)}, removing them")
        data.drop(data.iloc[invalid_indices].index.values, inplace=True)
    else:
        print("no invalid OHLC rows found")
    return data.sort_index()


def feature_engineering(data):
    data["Daily Return"] = data["Close"].pct_change()
    data["High Low Range"] = data["High"] - data["Low"]
    data["Open Close Range"] = (data["Close"] - data["Open"]).abs()
    data = calculate_volatility(data, 5)
    data.drop(data.index[:5],inplace=True)
    print('First 5 NaN Values Removed Cause the volatility')
    return data[['Open', 'High', 'Low', 'Close', "Daily Return", 'Volume', "Volatility", "Open Close Range", "High Low Range"]]


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
    return data.sort_index()


def preprocess_pipeline(data, save_dir = os.getcwd()):
    data = data.copy()

    num_duplicates = data['Date'].duplicated().sum()
    if num_duplicates:
        data['Date'].drop_duplicates(inplace=True)
        print(f'Warning: There is {num_duplicates} Data Duplicates, Removed Successfully!')

    data['Volume'] = data['Volume'].apply(lambda x: float(x.replace(',','')))

    before = len(data)

    data = remove_negative_prices(data)
    data = remove_invalid_ohlc(data)

    num_columns = len(data.columns)
    data = feature_engineering(data)
    print(f"Feature Engineered Successfully, Columns Before: {num_columns}, Columns After: {len(data.columns)}")

    data = remove_return_outliers(data)

    print(f"rows before: {before}, rows after: {len(data)}, total removed: {before - len(data)}")

    if save_dir:
        data.to_csv(os.path.join(save_dir,'data_processed.csv'), index=False)
    return data

def split_data(data,train_year,valid_year,test_year=None):
    train_set = data.loc[:str(train_year)].values
    valid_set = data.loc[str(train_year+1):str(valid_year)].values

    if test_year is None:
        test_year = data['Date'].dt.year.iloc[-1]

    test_set = data.loc[str(valid_year+1):str(test_year)].values
    print('Splitted Successfully to train/valid/test splits!')
    return train_set, valid_set, test_set


def norm_transform(train_set, valid_set, test_set):
    train_set_scaled = scaler.fit_transform(train_set)
    valid_set_scaled = scaler.transform(valid_set)
    test_set_scaled = scaler.transform(test_set)
    print('Normalization Transfromed Successfully!')
    return train_set_scaled, valid_set_scaled, test_set_scaled


def inverse_target(scaler, scaled_values, target_idx, num_features=None):
    scaled_values = np.asarray(scaled_values).reshape(-1, 1)

    if scaler.n_features_in_ == 1:
        return scaler.inverse_transform(scaled_values).flatten()

    if num_features is None:
        num_features = scaler.n_features_in_

    dummy = np.zeros((len(scaled_values), num_features))
    dummy[:, target_idx] = scaled_values.flatten()

    inversed = scaler.inverse_transform(dummy)

    return inversed[:, target_idx]


def create_sequences(data_set_scaled, target_idx, timesteps=60):
    X , y = [], []
    for i in range(timesteps, len(data_set_scaled)):
        X.append(data_set_scaled[i - timesteps:i])
        y.append(data_set_scaled[i,target_idx])

    print(f'Sequences Created Successfully, Sequences Counts: {len(X)}')
    return np.array(X), np.array(y)


def reshape_data(X_data, timesteps, num_features):
    return X_data.reshape(-1,timesteps,num_features)


def prepare_model_data(data,target_index,year_splits:list,timesteps,num_features, save_dir=os.getcwd()):
    train_set, valid_set, test_set = split_data(data,year_splits[0],year_splits[1],year_splits[2])

    train_set_scaled, valid_set_scaled, test_set_scaled = norm_transform(train_set, valid_set, test_set)

    X_train , y_train = create_sequences(train_set_scaled,target_index,timesteps)

    valid_input = np.concat([train_set_scaled[-timesteps:],valid_set_scaled])
    X_valid , y_valid = create_sequences(valid_input,target_index,timesteps)

    test_input = np.concat([valid_set_scaled[-timesteps:],test_set_scaled])
    X_test , y_test = create_sequences(test_input,target_index,timesteps)

    X_train, X_valid, X_test = (reshape_data(X_train, timesteps, num_features), 
                                reshape_data(X_valid, timesteps, num_features), 
                                reshape_data(X_test, timesteps, num_features))


    all_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_valid': X_valid,
        'y_valid': y_valid,  
        'X_test': X_test,
        'y_test': y_test
    }

    with open(os.path.join(save_dir,'all_model_data.pkl'), 'wb') as f:
        pickle.dump(all_data, f)

    with open(os.path.join(save_dir,'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print("Scaler & All data successfully saved in a single file!")

    return (
        all_data['X_train'], all_data['y_train'],
        all_data['X_valid'], all_data['y_valid'],
        all_data['X_test'], all_data['y_test']
    )