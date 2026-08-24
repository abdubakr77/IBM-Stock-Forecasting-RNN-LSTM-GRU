"""Train the RNN, LSTM, and GRU models and save them to the models/ directory."""

import os

def train_model(models, train_list, valid_list, epochs, batch_size, callbacks, save_dir=os.getcwd()):

    all_trained_models = {}
    all_models_history = {}

    for model in models:

        model_name = model.layers[0].name.upper()

        print(f"{model_name} Is Training...")

        history = model.fit(train_list[0],train_list[1],
                            validation_data=(valid_list[0],valid_list[1]),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[callbacks],
                            shuffle=False)

        model.save(os.path.join(save_dir,f"trained_model_{model_name}.keras"))
        all_trained_models[model_name] = model
        all_models_history[f"{model_name}_history"] = history
        
        print(f'\n{model_name} Saved Successfully!')

        print('\n\n')

    return all_trained_models , all_models_history
