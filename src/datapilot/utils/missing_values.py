# imports

import pandas as pd
from typing import Optional


def process_missing_values(
    df: pd.DataFrame,               # входной DataFrame
    threshold: float = 0.011,        # порог доли пропусков (по умолчанию 0.011%)
) -> pd.DataFrame:  
    """
    Обрабатывает пропуски в данных:
    - Если пропусков < threshold → удаляет строки
    - Если пропусков >= threshold → заполняет (mean для числовых, mode для категориальных)
    
    Выводит подробную статистику обработки.
    """
    
    print("=== АНАЛИЗ ИСХОДНЫХ ДАННЫХ ===")
    print(f"Форма данных: {df.shape}")
    
    # Анализ пропусков до обработки
    missing_before = df.isnull().sum().sum()
    print(f"Всего пропусков: {missing_before}")
    print(f"Порог: {threshold*100:.2f}%")
    
    missing_by_col = df.isnull().sum()
    if missing_by_col.sum() > 0:
        print("\nПропуски по колонкам:")
        for col in df.columns:
            if missing_by_col[col] > 0:
                pct = (missing_by_col[col] / len(df) * 100)
                # Определяем тип данных
                data_type = "numeric" if df[col].dtype in ["float64", "int64"] else "categorial"
                action = "удалим строки" if pct < threshold * 100 else "заменим значения"
                print(f"  {col} ({data_type}): {missing_by_col[col]} ({pct:.2f}%) → {action}")
    else:
        print("Пропусков не найдено!")
    
    # -----------------------
    # Проверка аргументов
    # -----------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df должен быть pandas DataFrame")
    if not 0 < threshold < 1:
        raise ValueError("threshold должен быть между 0 и 1")
    
    # -----------------------
    # Создаём копию DataFrame
    # -----------------------
    df_processed = df.copy()
    filled_count = 0  # счетчик замен
    
    print("\n=== ОБРАБОТКА ===")
    # -----------------------
    # Перебираем все колонки
    # -----------------------
    for col in df_processed.columns:
        missing_count = df_processed[col].isna().sum()  # количество пропусков
        missing_ratio = missing_count / len(df_processed)  # доля пропусков в колонке

        if missing_ratio > 0:
            if missing_ratio < threshold:
                # мало пропусков → удаляем строки с пропусками
                df_processed = df_processed.dropna(subset=[col])
                print(f"  {col}: {missing_ratio*100:.2f}% пропусков → удалили строки")
            else:
                # много пропусков → заполняем
                filled_count += missing_count  # добавляем к счетчику замен
                if df_processed[col].dtype in ["float64", "int64"]:  # числовая колонка
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                    print(f"  {col}: {missing_ratio*100:.2f}% пропусков → заменили {missing_count}")
                else:  # категориальная колонка
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                    print(f"  {col}: {missing_ratio*100:.2f}% пропусков → заменили {missing_count}")
    
    # -----------------------
    # Итоговый отчёт
    # -----------------------
    print("\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Исходная форма: {df.shape}")
    print(f"Новая форма: {df_processed.shape}")
    print(f"Удалено строк: {df.shape[0] - df_processed.shape[0]}")
    print(f"Заменено значений: {filled_count}")
    
    return df_processed