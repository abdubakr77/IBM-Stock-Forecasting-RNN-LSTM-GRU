# IBM Stock Forecasting: RNN vs LSTM vs GRU

A comparative study of Vanilla RNN, LSTM, and GRU architectures for forecasting IBM's daily closing stock price, using historical data from 1980 to 2025. Built with Keras and TensorFlow.

## Overview

This project trains and compares three recurrent neural network architectures on the same task: predicting IBM's next Close price from a rolling window of past trading days. The goal is not just to build one working model, but to see how the three architectures differ in accuracy and training behavior on the same data, same features, and same splits.

The pipeline covers the full workflow: exploratory data analysis, cleaning and feature engineering, sequence preparation, training each model separately, and a final notebook that loads all three trained models and compares them side by side.

## Dataset

IBM daily stock data from January 1980 to July 2025, sourced from Yahoo Finance. Includes Open, High, Low, Close, Volume, Dividends, and Stock Splits, along with an Adjusted Close column computed using CRSP-style adjustment standards.

The models in this project are trained to predict the raw Close price, not the Adjusted Close, since the goal is forecasting the actual price movement rather than a dividend and split adjusted historical series.

## Project Structure

```
IBM-Stock-Forecasting-RNN-LSTM-GRU/
│
├── data/
│   ├── raw/                        # Original dataset
│   └── processed/                  # Cleaned data, scaler, and prepared sequences
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_data_preprocessing.ipynb # Cleaning, feature engineering, sequence prep
│   ├── 03_vanilla_rnn.ipynb        # Vanilla RNN training
│   ├── 04_lstm.ipynb               # LSTM training
│   ├── 05_gru.ipynb                # GRU training
│   └── 06_models_comparison.ipynb  # Side by side comparison of all three models
│
├── src/
│   ├── data_preprocessing.py       # Cleaning, feature engineering, scaling, sequence creation
│   ├── model_rnn.py                # Vanilla RNN architecture
│   ├── model_lstm.py               # LSTM architecture
│   ├── model_gru.py                # GRU architecture
│   ├── train.py                    # Shared training function used by all three models
│   ├── evaluate.py                 # Evaluation, metrics, and comparison plots
│   └── utils.py                    # Shared helper functions (data checks, loading, etc)
│
├── models/                         # Saved trained models
├── results/
│   ├── metrics_comparisons.csv     # MSE / RMSE / MAE / MAPE for all three models
│   └── plots/                      # Loss curves, predicted vs true plots, comparison charts
│
├── requirements.txt
└── README.md
```

## Pipeline

**1. EDA (`01_eda.ipynb`)**
Checks the raw data for missing values, duplicates, invalid OHLC rows, negative prices, date gaps, correlation between columns, and outliers. Also looks at bullish versus bearish days, volatility over time, and daily return distribution.

**2. Preprocessing (`02_data_preprocessing.ipynb`)**
Runs the full cleaning pipeline: fixes the Volume column, removes invalid OHLC rows, removes negative or zero prices, removes extreme return outliers, and adds engineered features (Daily Return, Volatility, High Low Range, Open Close Range). All feature calculations use Close, not Adjusted Close, to keep the target consistent with the model's prediction goal. The data is then split by year into train, validation, and test sets, scaled with MinMaxScaler fitted only on the training set to avoid leakage, and turned into sliding window sequences of 90 timesteps.

**3. Model training (`03_vanilla_rnn.ipynb`, `04_lstm.ipynb`, `05_gru.ipynb`)**
Each notebook builds and trains one architecture on the same prepared sequences, with hyperparameters tuned separately per model (learning rate, weight decay, dropout, batch size). Training uses early stopping on validation loss to avoid overfitting. Each notebook saves its trained model and training history for later comparison.

**4. Comparison (`06_models_comparison.ipynb`)**
Loads all three trained models and their histories, plots loss and MSE curves for each, evaluates all three on the test set, inverse transforms predictions back to real dollar prices, plots predicted versus true prices per model, and generates a metrics comparison table and bar charts across MSE, RMSE, MAE, and MAPE.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the notebooks in order:

```
01_eda.ipynb
02_data_preprocessing.ipynb
03_vanilla_rnn.ipynb
04_lstm.ipynb
05_gru.ipynb
06_models_comparison.ipynb
```

Each notebook imports its logic from `src/`, so the notebooks themselves stay focused on running the pipeline and inspecting results rather than holding the implementation.

## Models Compared

- Vanilla RNN
- LSTM
- GRU

All three use the same input sequences (90 timesteps, 8 features) and predict the next Close price.

## Results

Final metrics (MSE, RMSE, MAE, MAPE) for all three models are saved in `results/metrics_comparisons.csv`. Training curves, predicted versus true price plots, and the metrics comparison chart are saved in `results/plots/`.