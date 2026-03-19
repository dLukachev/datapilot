# Это утилита для EDA класса для того, чтобы выявить нормальный способ
# заэмбедить признаки в фрейме

def embeding(uv, model_type, labels_columns, type_scaler):
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

    **type_scaler**:
    default StandartScaler()\n
    you can change to MinMaxScaler()

    **pr**:
    ...
    ...

    **pr**:
    ...
    ...
    """
    ...