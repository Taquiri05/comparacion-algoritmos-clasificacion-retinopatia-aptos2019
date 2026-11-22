# Fase 5, Paso 6: KNN. K y pesos elegidos por F1 medio en validacion cruzada (solo entrenamiento), con y sin PCA
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from preprocesamiento import COLS, construir_pipeline
from validacion import crear_cv

UMBRAL_PCA = 0.90                              # fijado de antemano (Paso 4)
tr = pd.read_csv("datos/train.csv")            # test.csv NO se abre
X = tr[COLS]
y = tr["y"]
cv = crear_cv()                                # mismos 5 pliegues de siempre

METRICAS = {"f1": "f1", "recall": "recall", "precision": "precision", "exactitud": "accuracy"}
VALORES_K = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
PESOS = ["uniform", "distance"]

t0 = time.time()
filas = []
for usar_pca in [False, True]:
    for pesos in PESOS:
        for K in VALORES_K:
            # log -> z-score -> [PCA 90 %] -> KNN (distancia euclidiana)
            pipe = construir_pipeline(KNeighborsClassifier(n_neighbors=K, weights=pesos),
                                      usar_pca=usar_pca, k=UMBRAL_PCA if usar_pca else None)
            r = cross_validate(pipe, X, y, cv=cv, scoring=METRICAS, n_jobs=-1)
            fila = {"pca": "con" if usar_pca else "sin", "pesos": pesos, "K": K}
            for nombre in METRICAS:
                v = r[f"test_{nombre}"]
                fila[f"{nombre}_media"] = v.mean()
                fila[f"{nombre}_desv"] = v.std()
            filas.append(fila)

tabla = pd.DataFrame(filas)
tabla.to_csv("resultados/fase5_knn.csv", index=False)

print(f"KNN (validacion cruzada de 5 pliegues, solo entrenamiento) - {time.time()-t0:.0f} s\n")
print("F1 medio por K")
ancho = tabla.pivot_table(index="K", columns=["pca", "pesos"], values="f1_media")
print(ancho.round(4).to_string())

print("\nMejor configuracion de cada grupo (por F1 medio):")
for (p, w), g in tabla.groupby(["pca", "pesos"]):
    b = g.loc[g["f1_media"].idxmax()]
    print(f"  {p} PCA, {w:<8}: K={int(b['K']):<3} F1 {b['f1_media']:.4f} (+-{b['f1_desv']:.4f}) | "
          f"recall {b['recall_media']:.4f} | precision {b['precision_media']:.4f} | exactitud {b['exactitud_media']:.4f}")

b = tabla.loc[tabla["f1_media"].idxmax()]
print(f"\nMejor global: {b['pca']} PCA, pesos {b['pesos']}, K={int(b['K'])}, F1 {b['f1_media']:.4f}")
print("Guardado: resultados/fase5_knn.csv")
