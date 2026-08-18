# IBM Stock Forecasting: RNN vs LSTM vs GRU

A comparative study of Vanilla RNN, LSTM, and GRU architectures for forecasting IBM's adjusted closing stock price using historical data from 1980 to 2025. Built with Keras/TensorFlow.

## Project Structure

```
IBM-Stock-Forecasting-RNN-LSTM-GRU/
│
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned and scaled data
│
├── notebooks/
├── 01_eda.ipynb
├── 02_data_preprocessing.ipynb
├── 03_vanilla_rnn.ipynb
├── 04_lstm.ipynb
├── 05_gru.ipynb
└── 06_models_comparison.ipynb
│
├── src/
│   ├── data_preprocessing.py # Data cleaning, scaling, sequence creation
│   ├── model_rnn.py          # Vanilla RNN architecture
│   ├── model_lstm.py         # LSTM architecture
│   ├── model_gru.py          # GRU architecture
│   ├── train.py               # Training script for all models
│   ├── evaluate.py            # Evaluation and comparison script
│   └── utils.py                # Shared helper functions
│
├── models/                    # Saved trained models
├── results/
│   ├── metrics_comparison.csv # RMSE / MAE / MAPE comparison
│   └── plots/                  # Prediction vs actual plots
│
├── requirements.txt
└── README.md
```

## Dataset

IBM daily stock data (1980–2025) sourced from Yahoo Finance, including an Adjusted Close column computed using CRSP-style adjustment standards for dividends and stock splits.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/train.py
python src/evaluate.py
```

## Models Compared

- Vanilla RNN
- LSTM
- GRU

## Results

Model performance metrics (RMSE, MAE, MAPE) are saved in `results/metrics_comparison.csv`, with visual comparisons in `results/plots/`.
