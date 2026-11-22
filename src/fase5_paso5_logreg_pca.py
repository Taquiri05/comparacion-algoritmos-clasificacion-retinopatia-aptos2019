# Fase 5, Paso 5: Regresion logistica CON PCA (umbral 90 % de varianza) y comparacion con el Paso 3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from preprocesamiento import COLS, construir_pipeline
from validacion import crear_cv

UMBRAL_PCA = 0.90                              # fijado de antemano (Paso 4)
tr = pd.read_csv("datos/train.csv")            # test.csv NO se abre
X = tr[COLS]
y = tr["y"]
cv = crear_cv()                                # mismos 5 pliegues de siempre

METRICAS = {"f1": "f1", "recall": "recall", "precision": "precision", "exactitud": "accuracy"}
VALORES_C = [0.01, 0.1, 1, 10, 100]

filas = []
for C in VALORES_C:
    # log -> z-score -> PCA (k elegido en cada pliegue con 90 %) -> regresion logistica
    pipe = construir_pipeline(LogisticRegression(C=C, max_iter=1000), usar_pca=True, k=UMBRAL_PCA)
    r = cross_validate(pipe, X, y, cv=cv, scoring=METRICAS, return_estimator=True)
    fila = {"C": C, "k_por_pliegue": [int(e.named_steps["pca"].n_components_) for e in r["estimator"]]}
    for nombre in METRICAS:
        v = r[f"test_{nombre}"]
        fila[f"{nombre}_media"] = v.mean()
        fila[f"{nombre}_desv"] = v.std()
    filas.append(fila)

tabla = pd.DataFrame(filas)
mejor = tabla.loc[tabla["f1_media"].idxmax(), "C"]

print("Regresion logistica CON PCA (validacion cruzada de 5 pliegues, solo entrenamiento)\n")
for _, f in tabla.iterrows():
    marca = "  <-- mejor F1" if f["C"] == mejor else ""
    print(f"C={f['C']:<6} F1 {f['f1_media']:.4f} (+-{f['f1_desv']:.4f}) | "
          f"recall {f['recall_media']:.4f} | precision {f['precision_media']:.4f} | "
          f"exactitud {f['exactitud_media']:.4f} | k={f['k_por_pliegue']}{marca}")

# Comparacion con el Paso 3 (sin PCA), mismos C y mismos pliegues
sin = pd.read_csv("resultados/fase5_logreg_sin_pca.csv")
print("\nComparacion (F1 medio):")
print(f"{'C':<7}{'sin PCA':>10}{'con PCA':>10}{'diferencia':>12}")
for a, b in zip(sin.itertuples(), tabla.itertuples()):
    print(f"{a.C:<7}{a.f1_media:>10.4f}{b.f1_media:>10.4f}{b.f1_media - a.f1_media:>+12.4f}")

tabla.to_csv("resultados/fase5_logreg_con_pca.csv", index=False)
print("\nGuardado: resultados/fase5_logreg_con_pca.csv")
