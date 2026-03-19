from typing import Literal
from pandas import DataFrame

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# Это утилита для EDA класса для того, чтобы выявить нормальный способ
# препроцессинка признаков в фрейме

def preprocessing(df: DataFrame | None = None,
                  uv: int = 5, 
                  model_type: Literal['linear', 'tree'] = 'linear',
                  scaler_columns: list[str] | None = None, 
                  labels_columns: list[str] | None = None, 
                  type_scaler: Literal['minmax', 'standart'] = 'standart'):
    """
    **uv**:
    unique values -> Up to what number of unique values in a column's\n
    data frame are considered categorical?

    **model_type**:
    linear or tree\n
    other not supported

    **labels_columns**:
    Если значения столбца можно сделать в виде числа, которые так же
    отражают их степень веса, то их необходимо указать.
    Например: уровень образования бакалавр -> 1, средне-специальное -> 0

    **scaler_columns**:
    Какие колонки необходимо заскейлить

    **type_scaler**:
    default StandartScaler()\n
    you can change to MinMaxScaler()
    """
    if df is None:
        raise ValueError("df must be provided")
    
    df = df.copy()

    if target is not None and target not in df.columns:
        raise ValueError("target column not in dataframe")
    
    categorical_cols = []
    numeric_cols = []

    # --- 1. Определяем типы признаков ---
    for col in df.columns:
        if col == target:
            continue

        if df[col].nunique() <= uv:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    # --- 2. Labels columns ---
    if labels_columns is not None and isinstance(labels_columns, list):
        for col in labels_columns:
            encoder = LabelEncoder()
            if col in df.columns:
                df[col] = encoder.fit_transform(df[col]) # type: ignore

    # --- 3. One-Hot для линейных моделей ---
    if model_type == "linear" and categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # --- 4. Scaling ---
    if model_type == "linear" and scaler_columns:
        scaler = StandardScaler() if type_scaler == "standart" else MinMaxScaler()

        cols_to_scale = [col for col in df.columns if col != target and col in scaler_columns]

        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df