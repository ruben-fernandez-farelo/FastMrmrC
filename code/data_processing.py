import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score

# Almacena los datos globalmente
data_features = None

def load_data(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Carga datos desde un archivo CSV.

    Parámetros:
        dataset_name (str): Nombre del dataset sin la extensión.

    Retorna:
        tuple: (x, y, gene_names)
    """
    raw_df = pd.read_csv(f"./data/{dataset_name}.csv")
    x, y = raw_df.iloc[:, 1:-1], raw_df.iloc[:, -1]
    y = np.logical_not(preprocessing.LabelEncoder().fit(y).transform(y)).astype(int)
    gene_names = raw_df.iloc[:, 0]
    return x, y, gene_names

def store_data_features(x: pd.DataFrame) -> None:
    """
    Almacena las características globalmente para su uso posterior.

    Parámetros:
        x (pd.DataFrame): DataFrame con las características.
    """
    global data_features
    data_features = x.copy()  # Evita modificar x accidentalmente

def get_data_features(indices) -> pd.DataFrame:
    """
    Obtiene ejemplos del dataset usando índices.

    Parámetros:
        indices (list): Lista de índices de las filas a recuperar.

    Retorna:
        pd.DataFrame: Subconjunto del dataset.
    """
    global data_features

    if data_features is None:
        raise ValueError("Error: data_features no ha sido inicializado. Llama a store_data_features primero.")

    if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
        raise ValueError(f"Índices inválidos para .iloc: {indices}")

    if max(indices) >= len(data_features):
        raise IndexError(f"Índice fuera de rango. Máximo permitido: {len(data_features) - 1}")

    return data_features.iloc[indices]

def generate_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    binary_threshold: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera características para los datos de entrenamiento y prueba.

    Parámetros:
        - x_train (pd.DataFrame): DataFrame con características de entrenamiento.
        - x_test (pd.DataFrame): DataFrame con características de prueba.
        - binary_threshold (float): Umbral para seleccionar características binarias.

    Retorna:
        - x_train_temp (pd.DataFrame): Características transformadas para entrenamiento.
        - x_test_temp (pd.DataFrame): Características transformadas para prueba.
    """
    # Verificar que los datos son DataFrames válidos
    if not isinstance(x_train, pd.DataFrame) or not isinstance(x_test, pd.DataFrame):
        raise TypeError("x_train y x_test deben ser DataFrames de pandas.")

    # Filtrar columnas en base al umbral de media
    valid_features = x_train.columns[x_train.mean() >= binary_threshold]

    # Mantener las mismas columnas en ambos conjuntos
    x_train_filtered = x_train[valid_features]
    x_test_filtered = x_test[valid_features]

    return x_train_filtered, x_test_filtered
