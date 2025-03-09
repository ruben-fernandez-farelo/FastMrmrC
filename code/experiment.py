import numpy as np
import pandas as pd
from typing import Literal, Union
from sklearn.model_selection import StratifiedKFold
import neptune
from code.model_training import train_a_model
from code.data_processing import load_data, store_data_features, get_data_features, generate_features
from code.hyperparameter_tuning import grid_search_hyperparams
from code.neptune_utils import upload_preds_to_neptune
import code.metrics as metrics
import code.pu_learning as pul


def run_experiment(
    dataset: str = None,
    classifier: str = None,
    pul_num_features: int = 0,
    pu_learning: Union[Literal["similarity", "threshold"], bool] = False,
    fast_mrmr: bool = False, # creo que esto es lo que falta cambiar y poner algo parecido a lo de pu_learning
    fast_mrmr_k: int = 0,
    search_space: dict = None,
    random_state: int = 42,
    neptune_run: Union[neptune.Run, None] = None,
):
    """Run a single experiment."""

    random_state = int(random_state)

    x, y, gene_names = load_data(dataset)


    pul.compute_pairwise_jaccard_measures(x)  # Keep this line for future use
    x = store_data_features(x)

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    experiment_preds, experiment_metrics = [], []

    for k, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):
        #x_train, x_test = x[train_idx], x[test_idx]  #TODO : x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]  # Usar `.iloc[]` para mantener DataFrame 

        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]         

        y_train, y_test = y[train_idx], y[test_idx]
        
   
        # Pass subsetted data directly (not indices)
        best_params = grid_search_hyperparams(
            x_train, 
            y_train,
            classifier=classifier,
            random_state=random_state,
            pu_learning=pu_learning,
            fast_mrmr=fast_mrmr,
            pul_num_features=pul_num_features,
            search_space=search_space,
            neptune_run=neptune_run,
        )

        pred_test = train_a_model(
            x_train,  
            y_train,
            x_test,   
            classifier=classifier,
            random_state=random_state,
            pu_learning=pu_learning,
            pul_num_features=pul_num_features,
            pul_k=best_params["pu_k"],
            pul_t=best_params["pu_t"],
            fast_mrmr=fast_mrmr,
            fast_mrmr_k=fast_mrmr_k,
        )

        experiment_preds += zip(test_idx, gene_names[test_idx], pred_test)
        fold_metrics = metrics.log_metrics(
            y_test, pred_test, neptune_run=neptune_run, run_number=random_state, fold=k
        )
        experiment_metrics.append(fold_metrics)

    experiment_preds = pd.DataFrame(experiment_preds, columns=["id", "gene", "prob"])

    if neptune_run:
        upload_preds_to_neptune(
            preds=experiment_preds,
            random_state=random_state,
            neptune_run=neptune_run,
        )

    for metric in experiment_metrics[0].keys():
        if neptune_run:
            neptune_run[f"metrics/run_{random_state}/avg/test/{metric}"] = np.mean(
                [fold_metrics[metric] for fold_metrics in experiment_metrics]
            )
        else:
            print(
                f"metrics/run_{random_state}/avg/test/{metric}: {np.mean([fold_metrics[metric] for fold_metrics in experiment_metrics])}"
            )

    return experiment_metrics, experiment_preds