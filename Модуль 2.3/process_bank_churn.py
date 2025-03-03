import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, List, Optional

def remove_unused_columns(df: pd.DataFrame, cols_to_remove: List[str]) -> pd.DataFrame:
    """
    Видаляє зазначені колонки з DataFrame.
    
    Args:
        df: Вхідний DataFrame.
        cols_to_remove: Список назв колонок, які потрібно видалити.
        
    Returns:
        DataFrame без вказаних колонок.
    """
    return df.drop(columns=cols_to_remove, errors='ignore')

def split_dataset(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Розбиває дані на тренувальну і валідаційну вибірки.
    
    Args:
        df: Вхідний DataFrame.
        target_col: Назва колонки з цільовою змінною.
        test_size: Частка даних для валідації.
        random_state: Початкове значення для генератора випадкових чисел.
        
    Returns:
        Кортеж з (X_train, y_train, X_val, y_val).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, y_train, X_val, y_val

def encode_categorical(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    Застосовує one-hot encoding до категоріальних колонок.
    
    Args:
        X_train: Тренувальні дані.
        X_val: Валідаційні дані.
        categorical_cols: Список назв категоріальних колонок.
        
    Returns:
        Кортеж з (X_train_encoded, X_val_encoded, fitted OneHotEncoder).
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Фітимо енкодер на тренувальних даних
    encoded_train = encoder.fit_transform(X_train[categorical_cols])
    encoded_val = encoder.transform(X_val[categorical_cols])
    
    # Створюємо DataFrame з новими ознаками
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df_train_encoded = pd.DataFrame(encoded_train, index=X_train.index, columns=encoded_cols)
    df_val_encoded = pd.DataFrame(encoded_val, index=X_val.index, columns=encoded_cols)
    
    # Прибираємо оригінальні категоріальні колонки та додаємо зашифровані
    X_train = X_train.drop(columns=categorical_cols)
    X_val = X_val.drop(columns=categorical_cols)
    X_train = pd.concat([X_train, df_train_encoded], axis=1)
    X_val = pd.concat([X_val, df_val_encoded], axis=1)
    
    return X_train, X_val, encoder

def scale_numeric(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Масштабує числові колонки за допомогою StandardScaler.
    
    Args:
        X_train: Тренувальні дані.
        X_val: Валідаційні дані.
        numeric_cols: Список назв числових колонок.
        
    Returns:
        Кортеж з (X_train_scaled, X_val_scaled, fitted StandardScaler).
    """
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    
    return X_train_scaled, X_val_scaled, scaler

def preprocess_data(
    raw_df: pd.DataFrame, 
    target_col: str = "Exited",
    scaler_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[StandardScaler], OneHotEncoder]:
    """
    Головна функція для попередньої обробки даних.
    
    Виконує наступні кроки:
      - Видаляє непотрібні колонки (наприклад, 'Surname').
      - Розбиває дані на тренувальну і валідаційну вибірки.
      - Виконує one-hot encoding категоріальних ознак.
      - Опціонально масштабує числові ознаки.
    
    Args:
        raw_df: Вхідний DataFrame з сирими даними.
        target_col: Назва колонки з цільовою змінною.
        scaler_numeric: Чи застосовувати масштабування числових ознак.
        
    Returns:
        Кортеж з:
         - X_train: Тренувальні ознаки.
         - train_targets: Цільова змінна для тренування.
         - X_val: Валідаційні ознаки.
         - val_targets: Цільова змінна для валідації.
         - input_cols: Список назв колонок ознак після обробки.
         - scaler: Фітнений StandardScaler або None.
         - encoder: Фітнений OneHotEncoder.
    """
    # Крок 1: Видаляємо непотрібні колонки
    df = remove_unused_columns(raw_df, cols_to_remove=["Surname"])
    
    # Припустимо, що категоріальні колонки – це 'Geography' та 'Gender'
    categorical_cols = ["Geography", "Gender"]
    # Числові колонки – всі, крім категоріальних і цільової
    numeric_cols = [col for col in df.columns if col not in categorical_cols + [target_col]]
    
    # Крок 2: Розбиття даних
    X_train, train_targets, X_val, val_targets = split_dataset(df, target_col=target_col)
    
    # Крок 3: Обробка категоріальних ознак
    X_train, X_val, encoder = encode_categorical(X_train, X_val, categorical_cols)
    
    # Крок 4: Опціональне масштабування числових даних
    scaler = None
    if scaler_numeric and numeric_cols:
        X_train, X_val, scaler = scale_numeric(X_train, X_val, numeric_cols)
    
    input_cols = list(X_train.columns)
    
    return X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder

def preprocess_new_data(
    new_df: pd.DataFrame, 
    input_cols: List[str],
    encoder: OneHotEncoder,
    scaler: Optional[StandardScaler] = None,
    scaler_numeric: bool = True
) -> pd.DataFrame:
    """
    Обробляє нові дані, використовуючи вже навчений енкодер та скейлер.
    
    Args:
        new_df: Новий DataFrame з даними, що підлягають обробці.
        input_cols: Список колонок, який очікується після обробки.
        encoder: Фітнений OneHotEncoder для категоріальних ознак.
        scaler: Фітнений StandardScaler для числових ознак (якщо використовується).
        scaler_numeric: Чи застосовувати масштабування числових ознак.
        
    Returns:
        DataFrame з обробленими даними, що містить ті ж колонки, що й input_cols.
    """
    # Видаляємо непотрібні колонки, якщо вони є
    df = remove_unused_columns(new_df, cols_to_remove=["Surname"])
    
    # Визначаємо категоріальні та числові колонки (припускаємо, що вони ті ж, що й під час навчання)
    categorical_cols = ["Geography", "Gender"]
    numeric_cols = [col for col in df.columns if col not in categorical_cols]
    
    # Обробка категоріальних ознак: застосовуємо вже навчений енкодер
    encoded_vals = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded_vals, index=df.index, columns=encoded_cols)
    
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, df_encoded], axis=1)
    
    # Масштабування числових даних, якщо потрібно
    if scaler_numeric and scaler is not None and numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # Забезпечуємо відповідність колонкам
    # Якщо якихось колонок не вистачає, додаємо їх з нулями
    for col in input_cols:
        if col not in df.columns:
            df[col] = 0.0
    # Переставляємо колонки у потрібному порядку
    df = df[input_cols]
    
    return df
