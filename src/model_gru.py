"""Build and compile the GRU model architecture."""

from keras.models import Sequential,Model
from keras.layers import Dense,GRU,Dropout,Input,Attention,MultiHeadAttention
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping

def build_gru(timesteps, num_features,dropout=0,lr=0.0005,weight_decay=0.0005,patience=10, load_weights_path=None):
    model = Sequential([
        Input((timesteps, num_features)),
        GRU(128, return_sequences=True),
        Dropout(dropout),
        GRU(32),
        Dropout(dropout),
        Dense(1, activation='linear')
    ])

    optim = AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])
    model.summary()
    callbacks = EarlyStopping('val_loss', patience=patience, restore_best_weights=True)

    if load_weights_path: model.load_weights(load_weights_path)

    return model, callbacks


def build_attention_with_gru(timesteps, num_features,dropout=0,lr=0.0001,weight_decay=0.0001,patience=10, load_weights_path=None):

    inputs = Input((timesteps, num_features))

    x = GRU(128,return_sequences=True)(inputs)
    x = Dropout(dropout)(x)

    atten_out = Attention()([x,x])

    x = GRU(32)(atten_out)
    x = Dropout(dropout)(x)

    atten_out = Attention()([x,x])

    x = GRU(4)(atten_out)

    output = Dense(1,activation='linear')(x)

    model = Model(inputs,output)

    optim = AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])
    model.summary()
    callbacks = EarlyStopping('val_loss', patience=patience, restore_best_weights=True)

    if load_weights_path: model.load_weights(load_weights_path)

    return model, callbacks