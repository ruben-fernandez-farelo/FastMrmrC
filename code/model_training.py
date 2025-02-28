import time
from typing import Literal, Union
import neptune
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from code.data_processing import generate_features
from code.models import get_model, set_class_weights
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
import code.pu_learning as pul
import subprocess
import os

CV_INNER = 5

def cv_train_with_params(
    x_train,
    y_train,
    classifier,
    random_state,
    pu_learning=False,
    pul_num_features=None,
    pul_k=None,
    pul_t=None,
    fast_mrmr=False,  # Add fast_mrmr
    fast_mrmr_k=None,  # Add fast_mrmr_k
):
    # (Tu código de cv_train_with_params)
    inner_skf = StratifiedKFold(
        n_splits=CV_INNER, shuffle=True, random_state=random_state
    )

    score = []

    for _, (learn_idx, val_idx) in enumerate(inner_skf.split(x_train, y_train)):

        x_learn, x_val = x_train.iloc[learn_idx], x_train.iloc[val_idx]
        y_learn, y_val = y_train.iloc[learn_idx], y_train.iloc[val_idx]

        pred_val = train_a_model(
            x_learn,
            y_learn,
            x_val,
            classifier,
            random_state=random_state,
            pu_learning=pu_learning,
            pul_num_features=pul_num_features,
            pul_k=pul_k,
            pul_t=pul_t,
            fast_mrmr=fast_mrmr,  # Add fast_mrmr
            fast_mrmr_k=fast_mrmr_k,  # Add fast_mrmr_k
        )

        if pu_learning:
            score.append(f1_score(y_val, pred_val > 0.5))
        else:
            score.append(geometric_mean_score(y_val, pred_val > 0.5))

    return np.mean(score)

def train_a_model(
    x_train,
    y_train,
    x_test,
    classifier: Literal["CAT", "BRF", "XGB", "EEC"],
    random_state: int,
    pu_learning: str | bool = False,
    pul_num_features=None,
    pul_k: int = None,
    pul_t: float = None,
    fast_mrmr: bool = False,
    fast_mrmr_k: int = 0,
):
    if fast_mrmr:
        # Ejecutar fast-mrmr y capturar salida

        result = subprocess.run(
            "./fast-mrmr -a " + str(fast_mrmr_k),
            shell=True,
            capture_output=True,
            text=True,
            cwd="src_c"  # <-- Esto cambia el directorio de trabajo antes de ejecutar el comando
        )


        print(result.stdout)
        # Obtener los índices de las características seleccionadas
        selected_features_indices = list(map(int, result.stdout.strip().split(',')))

        # Seleccionar las columnas correspondientes en x_train y x_test
        selected_features = x_train.columns[selected_features_indices]
        x_train = x_train[selected_features]
        x_test = x_test[selected_features]
    

    if pu_learning:
        pul.feature_selection_jaccard(
            x_train,
            y_train,
            pul_num_features,
            classifier=classifier,
            random_state=random_state,
        )

        x_train, y_train = pul.select_reliable_negatives(
            x_train,
            y_train,
            pu_learning,
            pul_k,
            pul_t,
            random_state=random_state,
        )

    x_train, x_test = generate_features(x_train, x_test)

    model = get_model(classifier, random_state=random_state)

    if classifier == "CAT":
        model = set_class_weights(model, y_train)
        model.fit(x_train, y_train, verbose=0)
    elif classifier == "XGB":
        pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
        model.set_params(scale_pos_weight=pos_weight)
        model.fit(x_train, y_train, verbose=0)
    else:
        model.fit(x_train, y_train)

    probs = model.predict_proba(x_test)[:, 1]

    return probs