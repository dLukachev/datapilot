from typing import Literal, Iterable
import pandas as pd
import numpy as np

def remove_outliers(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    method: Literal["iqr", "zscore", "percentile"] = "iqr",
    iqr_k: float = 1.5,
    z_thresh: float = 3.0,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """
    Удаляет выбросы из DataFrame и возвращает очищенный DataFrame.

    Параметры
    ---------
    df : pd.DataFrame
        Исходный датафрейм.

    columns : Iterable[str] | None, по умолчанию None
        Список числовых колонок, в которых искать выбросы.
        Если None — берутся все числовые столбцы df.select_dtypes(include=[np.number]).

    method : {"iqr", "zscore", "percentile"}, по умолчанию "iqr"
        Способ обнаружения выбросов:
        - "iqr"        — выбросы за пределами [Q1 - k*IQR, Q3 + k*IQR],
                          где IQR = Q3 - Q1 и k = iqr_k.
        - "zscore"     — выбросы с |z| > z_thresh, где
                          z = (x - mean) / std.
        - "percentile" — выбросы вне диапазона
                          [quantile(lower_pct), quantile(upper_pct)].

    iqr_k : float, по умолчанию 1.5
        Множитель для IQR-метода.

    z_thresh : float, по умолчанию 3.0
        Порог для |z|-оценки при методе "zscore".

    lower_pct : float, по умолчанию 0.01
        Нижний процентиль (0..1) для метода "percentile".

    upper_pct : float, по умолчанию 0.99
        Верхний процентиль (0..1) для метода "percentile".

    Возвращает
    ----------
    pd.DataFrame
        Новый датафрейм, из которого удалены строки, содержащие выбросы
        хотя бы в одной из выбранных колонок.
    """
    if columns is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = list(columns)

    if not cols:
        return df.copy()

    mask = pd.Series(True, index=df.index)

    if method == "iqr":
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            low = q1 - iqr_k * iqr
            high = q3 + iqr_k * iqr
            mask &= df[col].between(low, high)
    elif method == "zscore":
        for col in cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                continue
            z = (df[col] - mean) / std
            mask &= z.abs() <= z_thresh
    elif method == "percentile":
        for col in cols:
            low = df[col].quantile(lower_pct)
            high = df[col].quantile(upper_pct)
            mask &= df[col].between(low, high)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return df[mask].copy()
