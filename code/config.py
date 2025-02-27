import argparse
from typing import Union

def read_config() -> argparse.Namespace:

    """
    Read the configuration parameters from the command line.

    Returns:
        dict: Configuration parameters
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--classifier", type=str, help="Classifier name")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--cv_outer", type=int, default=10, help="Outer cross-validation folds")
    parser.add_argument("--cv_inner", type=int, default=5, help="Inner cross-validation folds")
    parser.add_argument("--binary_threshold", type=float, default=0.005, help="Binary threshold")
    parser.add_argument("--pu_learning", type=str, default=False, help="PU learning")
    parser.add_argument("--pu_k", type=int, nargs="+", default=10, help="PU k")
    parser.add_argument("--pu_t", type=float, nargs="+", default=None, help="PU t values")
    parser.add_argument("--pul_num_features", type=int, default=0, help="PUL kNN number of features")
    parser.add_argument("--neptune", type=bool, default=False, help="Neptune logging")

    # FAST MRMR , lo de store true es para que se active el argumento, y el type es para que sea un entero
    parser.add_argument("--fast_mrmr", action="store_true", help="Enable Fast MRMR feature selection")  # Use action="store_true"
    parser.add_argument("--fast_mrmr_k", type=int, default=0, help="Number of features to select with Fast MRMR") # Required if fast_mrmr is True
    

    args = parser.parse_args()
    args = vars(args)

    return args
