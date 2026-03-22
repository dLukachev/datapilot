from numpy.typing import ArrayLike
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, roc_auc_score, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

class Metrics():

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def linear_report(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[tuple[float, float, float, float], tuple[str]]:
        """
        Only linear report. For classification use 'from sklearn.metrics import classification_report'
        
        **return** tuple((mae, rmse, mape, r2), (f'MAE: {mae}, RMSE: {rmse}, MAPE: {mape}, r2: {r2}',))
        """
        try:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            # roc_auc = roc_auc_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
        except Exception as e:
            raise e
        return ((mae, rmse, mape, r2), (f'MAE: {mae}, RMSE: {rmse}, MAPE: {mape}, r2: {r2}',))
    

    @staticmethod
    def linear_plot(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        figsize: tuple = (8, 6)
    ) -> None:
        """
        Scatter plot: y_true vs y_pred
        цвет = абсолютная ошибка
        """
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            if y_true.shape != y_pred.shape:
                raise ValueError("y_true and y_pred must have the same shape")

            if len(y_true) == 0:
                raise ValueError("Empty input arrays")

            # удаляем NaN
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            errors = y_pred - y_true
            abs_errors = np.abs(errors)

            plt.figure(figsize=figsize)

            scatter = plt.scatter(
                y_true,
                y_pred,
                c=abs_errors,
                alpha=0.7
            )

            # линия идеала
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())

            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                linestyle="--"
            )

            plt.xlabel("y_true")
            plt.ylabel("y_pred")
            plt.title("Prediction vs True")

            plt.colorbar(scatter, label="Absolute Error")

            plt.grid(True)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            raise ValueError(f"Error in linear_plot: {e}")