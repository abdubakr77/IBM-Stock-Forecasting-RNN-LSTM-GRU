"""Build and compile the Vanilla RNN model architecture."""

from keras.models import Sequential
from keras.layers import Dense,RNN,SimpleRNNCell,Dropout
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping
import os

def build_rnn(timesteps, num_features,dropout=0,lr=0.0005,weight_decay=0.0005,patience=10, load_weights_path=None):
    model = Sequential([
        RNN(SimpleRNNCell(64), return_sequences=True, input_shape=(timesteps, num_features)),
        Dropout(dropout),
        RNN(SimpleRNNCell(16)),
        Dropout(dropout),
        Dense(1, activation='linear')
    ])

    optim = AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])
    model.summary()

    callbacks = EarlyStopping('val_loss', patience=patience, restore_best_weights=True)

    if load_weights_path: model.load_weights(load_weights_path)

    return model, callbacks