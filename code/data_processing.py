import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score

data_features = None


def load_data(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from a CSV file.

    Parameters:
    dataset_path (str): The name of the dataset.

    Returns:
    tuple: A tuple containing the features (x) and the labels (y).
    """

    raw_df = pd.read_csv(f"./data/{dataset_name}.csv")
    x, y = raw_df.iloc[:, 1:-1], raw_df.iloc[:, -1]
    y = np.logical_not(preprocessing.LabelEncoder().fit(y).transform(y)).astype(int)
    gene_names = raw_df.iloc[:, 0]
    return x, y, gene_names


def store_data_features(x: pd.DataFrame) -> None:
    """
    Store the features of the data for later use.

    Parameters:
    x (pd.DataFrame): The features of the data.
    """
    global data_features
    data_features = x.copy()  # Asegurar que se mantiene como DataFrame
    
    #x = np.arange(len(x)) # x = np.arange(len(x))  #TODO: Esto está sobreescribiendo `x` con un array de numpy., CAMBIO: Comentar esta línea

    return x


def get_data_features(indices) -> pd.DataFrame:
    """
    Get the examples of the data with the specified indices.

    Parameters:
    indices (list): The indices of the examples to retrieve.

    Returns:
    pd.DataFrame: The examples with the specified indices.
    """
    return data_features.iloc[indices]

def generate_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate features for training and testing data based on specified parameters.

    Parameters:
        - x_train (pd.DataFrame): Training data features.
        - x_test (pd.DataFrame): Testing data features.
        - y_train (pd.Series): Training data labels.
        - y_test (pd.Series): Testing data labels.
        - params (dict): Dictionary containing parameters for feature generation.
        - random_state (int, optional): Random seed.
        - verbose (int, optional): Whether to print number and type of features used.
    Returns:
        - x_train_temp (pd.DataFrame): Transformed training data features.
        - x_test_temp (pd.DataFrame): Transformed testing data features.
    """
    
    return x_train, x_test