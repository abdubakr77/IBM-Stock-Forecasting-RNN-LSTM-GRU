"""Train the RNN, LSTM, and GRU models and save them to the models/ directory."""

import os
import pickle

def train_model(model, train_data, valid_data, epochs, batch_size, callbacks=None, model_name=None, save_dir=os.getcwd()):
    if model_name is None:
        model_name = model.layers[0].name.upper()

    print(f"{model_name} Is Training...")

    history = model.fit(
        train_data[0], train_data[1],
        validation_data=(valid_data[0], valid_data[1]),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks if isinstance(callbacks, list) else [callbacks] if callbacks else None,
        shuffle=False
    )

    model.save(os.path.join(save_dir, f"trained_model_{model_name}.keras"), overwrite=True)
    
    with open(os.path.join(save_dir, f"{model_name}_history.pkl"), "wb") as f:
        pickle.dump(history, f)

    print(f"\n{model_name} Saved Successfully!")

    return model, history
