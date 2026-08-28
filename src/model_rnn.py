"""Build and compile the Vanilla RNN model architecture."""

from keras.models import Sequential,Model
from keras.layers import Dense,RNN,SimpleRNNCell,Dropout,Input,Attention
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping

def build_rnn(timesteps, num_features,dropout=0,lr=0.0005,weight_decay=0.0005,patience=10, load_weights_path=None):
    model = Sequential([
        Input((timesteps, num_features)),
        RNN(SimpleRNNCell(64), return_sequences=True),
        RNN(SimpleRNNCell(32)),
        Dropout(dropout),
        Dense(1, activation='linear')
    ])

    optim = AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])
    model.summary()

    callbacks = EarlyStopping('val_loss', patience=patience, restore_best_weights=True,verbose=1)

    if load_weights_path: model.load_weights(load_weights_path)

    return model, callbacks


def build_attention_with_rnn(timesteps, num_features,dropout=0,lr=0.0001,weight_decay=0.0001,patience=10, load_weights_path=None):

    inputs = Input((timesteps, num_features))

    x = RNN(SimpleRNNCell(128),return_sequences=True)(inputs)
    x = Dropout(dropout)(x)

    atten_out = Attention()([x,x])

    x = RNN(SimpleRNNCell(32))(atten_out)
    x = Dropout(dropout)(x)

    output = Dense(1,activation='linear')(x)

    model = Model(inputs,output)

    optim = AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])
    model.summary()
    callbacks = EarlyStopping('val_loss', patience=patience, restore_best_weights=True,verbose=1)

    if load_weights_path: model.load_weights(load_weights_path)

    return model, callbacks