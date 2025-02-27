import subprocess
import pandas as pd
import tempfile
import os

def run_fast_mrmr(X, y, n_features, class_index=0):
    """
    Ejecuta fastMRMR desde Python
    
    Args:
        X (pd.DataFrame): Datos de entrada (muestras x características)
        y (pd.Series): Variable objetivo(genes)
        n_features (int): Número de características a seleccionar
        class_index (int): Índice de la columna de clase
    
    Returns:
        list: Índices de las características seleccionadas
    """
    # Crear un archivo temporal para guardar los datos
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Crear el archivo de datos
        data_file = os.path.join(tmp_dir, "data.csv")
        X["class"] = y
        X.to_csv(data_file, index=False)
        
        # Ejecutar fastMRMR
        output = subprocess.check_output(
            [
                "fast-mrmr",
                "-f", data_file,
                "-n", str(n_features),
                "-c", str(class_index),
            ]
        ).decode("utf-8")
        
        # Leer los índices de las características seleccionadas
        selected_features = [int(x) for x in output.strip().split(",")]
        
    return selected_features