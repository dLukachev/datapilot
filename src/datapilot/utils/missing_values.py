import pandas as pd

def process_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.011,        # порог доли пропусков (по умолчанию 0.011%)
) -> pd.DataFrame:  
    """
    Обрабатывает пропуски в данных:
    - Если пропусков < threshold → удаляет строки
    - Если пропусков >= threshold → заполняет (mean для числовых, mode для категориальных)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df должен быть pandas DataFrame")
    if not 0 < threshold < 1:
        raise ValueError("threshold должен быть между 0 и 1")

    df_processed = df.copy()

    # -----------------------
    # Перебираем все колонки
    # -----------------------
    for col in df_processed.columns:
        missing_count = df_processed[col].isna().sum()     # количество пропусков
        missing_ratio = missing_count / len(df_processed)  # доля пропусков в колонке

        if missing_ratio > 0:
            if missing_ratio < threshold:
                # мало пропусков → удаляем строки с пропусками
                df_processed = df_processed.dropna(subset=[col])
            else:
                # много пропусков → заполняем
                if df_processed[col].dtype in ["float64", "int64"]:  # числовая колонка
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                else:  # категориальная колонка
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    return df_processed