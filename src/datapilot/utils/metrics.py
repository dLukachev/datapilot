from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, roc_auc_score, mean_absolute_percentage_error, r2_score

class Metrics():

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def linear_report(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[float, float, float, float]:
        """
        Only linear report. For classification use 'from sklearn.metrics import classification_report'
        
        **return** tuple(mae, rmse, mape, r2)
        """
        try:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            # roc_auc = roc_auc_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
        except Exception as e:
            raise e
        return (mae, rmse, mape, r2)
    

    @staticmethod
    def linear_plot():
        """
        Принимает y-true и y-pred и рисует разброс ошибки на графике scatterplot
        """
        ...