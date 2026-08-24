"""Train the RNN, LSTM, and GRU models and save them to the models/ directory."""

import os

def train_model(model, train_list, valid_list, epochs, batch_size, callbacks, save_dir=os.getcwd()):

    history = model.fit(train_list[0],train_list[1],
                        validation_data=(valid_list[0],valid_list[1]),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[callbacks],
                        shuffle=False)

    model.save(save_dir)

    return model , history
