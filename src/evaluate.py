"""Evaluate trained models, compute metrics, and generate comparison plots."""

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os
import pandas as pd

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


def evaluate_all_models(models: list, X_test, y_test, save_dir=os.getcwd()):
    results = {}
    metrics_list = []

    for model in models:
        model_name = model.layers[0].name.upper()

        loss, mse = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test, verbose=0).flatten()

        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        print(f"{model_name} - loss: {loss:.4f}, mse: {mse:.4f}, rmse: {rmse:.4f}, mae: {mae:.4f}, mape: {mape:.2f}%")

        results[model_name] = {"loss": loss, "mse": mse, "rmse": rmse, "mae": mae, "mape": mape}
        metrics_list.append({"Model": model_name, "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape})

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(save_dir, "metrics_comparisons.csv"), index=False)
    print("metrics saved successfully")

    return results


def generate_comparison_plots(metrics_df, save_dir=os.getcwd()):
    metrics_cols = ["MSE", "RMSE", "MAE", "MAPE"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, col in zip(axes.flatten(), metrics_cols):
        ax.bar(metrics_df["Model"], metrics_df[col])
        ax.set_title(col)
        ax.set_xlabel("Model")
        ax.set_ylabel(col)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_comparison.png"))
    plt.show()