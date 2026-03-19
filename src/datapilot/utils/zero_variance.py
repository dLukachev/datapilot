# low_zero_variance.py

from typing import Optional
import pandas as pd
from pandas import DataFrame
from sklearn.feature_selection import VarianceThreshold


def low_zero_variance(
    df: DataFrame,
    target: Optional[str] = None,
    threshold: float = 2e-2,
) -> DataFrame:
    """
    Удаляет из df признаки с нулевой или низкой дисперсией.
    Если target указан, он не участвует в отборе и возвращается без изменений.

    По дефолту значние threshold = 2%
    
    threshold=0      -> только строго константные признаки
    threshold>0      -> zero- и low-variance признаки
    """
    # отделяем target, если он есть
    if target is not None and target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
    else:
        y = None
        X = df

    # sklearn работает и с DataFrame, но теряет имена колонок на выходе,
    # поэтому потом восстановим их вручную.
    selector = VarianceThreshold(threshold=threshold)
    X_reduced_array = selector.fit_transform(X)

    # оставшиеся признаки
    kept_cols = X.columns[selector.get_support()]

    # собираем обратно DataFrame
    X_reduced = pd.DataFrame(X_reduced_array, columns=kept_cols, index=df.index)

    if y is not None:
        X_reduced[target] = y

    return X_reduced
