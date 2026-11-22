# src\preprocesamiento.py
# Preprocesamiento compartido para la Fase 5 y 6: log(1+x) en 4 columnas -> StandardScaler -> [PCA] -> modelo
# Todo lo que aprende de los datos (media, desviacion, componentes) se ajusta solo con lo que reciba fit().
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA

COLS = ["area_rojas", "n_rojas", "area_exudados", "n_exudados", "densidad_vasos",
        "glcm_contraste", "glcm_homogeneidad", "glcm_energia", "glcm_correlacion"]
COLS_LOG = ["area_rojas", "n_rojas", "area_exudados", "n_exudados"]      # las 4 muy asimetricas
COLS_ORDEN_SALIDA = COLS_LOG + [c for c in COLS if c not in COLS_LOG]    # orden de columnas tras el ColumnTransformer

def construir_preprocesamiento():
    # Paso 1: ln(1+x) en 4 columnas (formula fija, no aprende nada). Paso 2: z-score (aprende mu y sigma)
    log = ColumnTransformer([("log1p", FunctionTransformer(np.log1p), COLS_LOG)], remainder="passthrough")
    return Pipeline([("log", log), ("escalador", StandardScaler())])

def construir_pipeline(modelo, usar_pca=False, k=None):
    # Pipeline completo: log -> StandardScaler -> [PCA(k)] -> modelo
    pasos = [("log", ColumnTransformer([("log1p", FunctionTransformer(np.log1p), COLS_LOG)], remainder="passthrough")),
             ("escalador", StandardScaler())]
    if usar_pca:
        pasos.append(("pca", PCA(n_components=k, random_state=42)))
    pasos.append(("modelo", modelo))
    return Pipeline(pasos)
