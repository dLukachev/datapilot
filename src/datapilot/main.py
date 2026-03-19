import pandas as pd
import numpy as np

from numpy.typing import ArrayLike
from pandas import DataFrame

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, roc_auc_score, mean_absolute_percentage_error, r2_score

import matplotlib.pyplot as plt

class EDA:
    def __init__(self, df: DataFrame) -> None:
        self.df = df.convert_dtypes()
        self.shape = self.df.shape
        self.describe = self.df.describe()
        self.unique = self.df.nunique()
        self.null_value = self.df.isna().sum()
        self.null_percent = (self.df.isna().mean() * 100).sort_values(ascending=False)
        self.numeric = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical = self.df.select_dtypes(include=['string', 'object', 'category']).columns.tolist()
        self.datetime = self.df.select_dtypes(include=['datetime']).columns.tolist()
        self.boolean = self.df.select_dtypes(include=['bool']).columns.tolist()
        self.corr = self._correlation(self.df)


    @classmethod
    def _correlation(cls, df: DataFrame) -> DataFrame:
        """
        df = all DataFrame. Dont df.corr()
        
        Only numeric type
        """
        df_numeric = df.select_dtypes(include=['number'])
        corr = df_numeric.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        filtered = (
            corr.where(mask)
                .stack()
                .reset_index()
        )

        filtered.columns = ["feature_1", "feature_2", "correlation"]

        filtered = filtered[
            (filtered["feature_1"] != filtered["feature_2"]) &
            (filtered["correlation"].abs() > 0.5)
        ]

        return filtered.sort_values(by="correlation", key=abs, ascending=False)


    def head(self, n: int = 5) -> DataFrame:
        return self.df.head(n)


    def missing_report(self):
        return pd.DataFrame({
            "missing_count": self.df.isna().sum(),
            "missing_percent": self.df.isna().mean() * 100
        }).sort_values("missing_percent", ascending=False)

    
    def info(self) -> dict:
        """Just show all info"""
        print(f'Shape -> {self.shape}')
        print()
        print(f'Describe:')
        print(self.describe)
        print()
        print(f'Unique values')
        print(self.unique)
        print()
        print(f'N/A values:')
        print(self.null_value)
        print()
        print(f'Feautures:')
        print(f'Categorical', self.categorical, sep=" -> ")
        print(f'Numeric', self.numeric, sep=" -> ")
        print(f'Datetime', self.datetime, sep=" -> ")
        print(f'Boolean', self.boolean, sep=" -> ")
        print()
        print('Correlation >50% :')
        print(self.corr)
        return {"shape": self.shape,
                "describe": self.describe,
                "null_values": self.null_value,
                "features": {
                    "categorical": self.categorical,
                    "numeric": self.numeric,
                    "datetime": self.datetime,
                    "boolean": self.boolean,
                },
                "correlation_gt_50_percent": self.corr}

    
    def distribution(self) -> None:
        """Shows the distribution of all features of the data frame"""
        cols = self.categorical + self.categorical + self.datetime + self.boolean
        n = len(cols)

        n_cols = 4
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            axes[i].hist(self.df[col].dropna(), bins=30)
            axes[i].set_title(col)

        for j in range(n, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


    def recommendations(self):
        """
        1. Missing values
        2. Zero variance
        3. Correlation
        4. Encoding (текущая активность)
        5. Emissions

        """
        ...

    
    def transform(self):
        """Автоматически, на основе рекомендаций, применяет изменения к DataFrame"""
        ...