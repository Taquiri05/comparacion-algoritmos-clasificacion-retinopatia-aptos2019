# Fase 5, Paso 3: Regresion logistica SIN PCA. Elegir C por F1 medio en validacion cruzada (solo entrenamiento)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from preprocesamiento import COLS, construir_pipeline
from validacion import crear_cv

tr = pd.read_csv("datos/train.csv")          # test.csv NO se abre
X = tr[COLS]
y = tr["y"]
cv = crear_cv()                               # mismos 5 pliegues de siempre

METRICAS = {"f1": "f1", "recall": "recall", "precision": "precision", "exactitud": "accuracy"}
VALORES_C = [0.01, 0.1, 1, 10, 100]

filas = []
for C in VALORES_C:
    # Pipeline completo dentro de cada pliegue: log -> z-score -> regresion logistica
    pipe = construir_pipeline(LogisticRegression(C=C, max_iter=1000))
    r = cross_validate(pipe, X, y, cv=cv, scoring=METRICAS)
    fila = {"C": C}
    for nombre in METRICAS:
        v = r[f"test_{nombre}"]
        fila[f"{nombre}_media"] = v.mean()
        fila[f"{nombre}_desv"] = v.std()      # desviacion poblacional (n = 5 pliegues)
    filas.append(fila)

tabla = pd.DataFrame(filas)
mejor = tabla.loc[tabla["f1_media"].idxmax(), "C"]

print("Regresion logistica sin PCA (validacion cruzada de 5 pliegues, solo entrenamiento)\n")
for _, f in tabla.iterrows():
    marca = "  <-- mejor F1" if f["C"] == mejor else ""
    print(f"C={f['C']:<6} F1 {f['f1_media']:.4f} (+-{f['f1_desv']:.4f}) | "
          f"recall {f['recall_media']:.4f} | precision {f['precision_media']:.4f} | "
          f"exactitud {f['exactitud_media']:.4f}{marca}")

tabla.to_csv("resultados/fase5_logreg_sin_pca.csv", index=False)
print("\nGuardado: resultados/fase5_logreg_sin_pca.csv")
