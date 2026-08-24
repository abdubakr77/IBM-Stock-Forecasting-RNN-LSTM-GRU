"""Evaluate trained models, compute metrics, and generate comparison plots."""

import matplotlib.pyplot as plt
import os

def show_all_history_plots(all_history:dict, save_dir=os.getcwd()):
    num_models = len(all_history)
    fig, axes = plt.subplots(num_models, 2, figsize=(14, 5 * num_models))

    if num_models == 1:
        axes = axes.reshape(1, 2)

    for row, (model_name, history) in enumerate(all_history.items()):
        history_dict = history.history

        axes[row, 0].plot(history_dict['loss'], label='Train Loss')
        axes[row, 0].plot(history_dict['val_loss'], label='Validation Loss')
        axes[row, 0].set_title(f'{model_name} - Loss Over Epochs')
        axes[row, 0].set_xlabel('Epoch')
        axes[row, 0].set_ylabel('Loss')
        axes[row, 0].legend()

        axes[row, 1].plot(history_dict['mse'], label='Train MSE')
        axes[row, 1].plot(history_dict['val_mse'], label='Validation MSE')
        axes[row, 1].set_title(f'{model_name} - MSE Over Epochs')
        axes[row, 1].set_xlabel('Epoch')
        axes[row, 1].set_ylabel('MSE')
        axes[row, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_models_loss_mse_comparison.png'))
    plt.show()


def evaluate_all_models(models:list, X_test, y_test):
    results = {}

    for model in models:
        model_name = model.layers[0].name.upper()
        loss, mse = model.evaluate(X_test, y_test, verbose=0)
        print(f"{model_name} - loss: {loss:.4f}, mse: {mse:.4f}")
        results[model_name] = {"loss": loss, "mse": mse}
    return results