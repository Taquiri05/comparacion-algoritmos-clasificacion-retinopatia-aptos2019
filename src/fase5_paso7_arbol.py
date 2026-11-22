# Fase 5, Paso 7: Arbol de decision. criterion, max_depth y min_samples_leaf por F1 medio en validacion cruzada
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from preprocesamiento import COLS, construir_pipeline
from validacion import crear_cv

UMBRAL_PCA = 0.90                              # fijado de antemano (Paso 4)
tr = pd.read_csv("datos/train.csv")            # test.csv NO se abre
X = tr[COLS]
y = tr["y"]
cv = crear_cv()                                # mismos 5 pliegues de siempre

METRICAS = {"f1": "f1", "recall": "recall", "precision": "precision", "exactitud": "accuracy"}
CRITERIOS = ["gini", "entropy"]
PROFUNDIDADES = [2, 3, 4, 5, 6, 8, 10]
HOJAS_MIN = [1, 5, 10, 20, 50]

t0 = time.time()
filas = []
for usar_pca in [False, True]:
    for crit in CRITERIOS:
        for prof in PROFUNDIDADES:
            for hojas in HOJAS_MIN:
                # log -> z-score -> [PCA 90 %] -> arbol (log y z-score no cambian las preguntas del arbol)
                modelo = DecisionTreeClassifier(criterion=crit, max_depth=prof,
                                                min_samples_leaf=hojas, random_state=42)
                pipe = construir_pipeline(modelo, usar_pca=usar_pca, k=UMBRAL_PCA if usar_pca else None)
                r = cross_validate(pipe, X, y, cv=cv, scoring=METRICAS, n_jobs=-1)
                fila = {"pca": "con" if usar_pca else "sin", "criterio": crit,
                        "max_depth": prof, "min_samples_leaf": hojas}
                for nombre in METRICAS:
                    v = r[f"test_{nombre}"]
                    fila[f"{nombre}_media"] = v.mean()
                    fila[f"{nombre}_desv"] = v.std()
                filas.append(fila)

tabla = pd.DataFrame(filas)
tabla.to_csv("resultados/fase5_arbol.csv", index=False)

print(f"Arbol de decision (validacion cruzada de 5 pliegues, solo entrenamiento) - {time.time()-t0:.0f} s\n")
print("Mejor F1 medio por profundidad (probando todos los criterios y hojas minimas)")
print(tabla.pivot_table(index="max_depth", columns="pca", values="f1_media", aggfunc="max").round(4).to_string())

print("\nMejor configuracion de cada grupo (por F1 medio):")
for (p, c), g in tabla.groupby(["pca", "criterio"]):
    b = g.loc[g["f1_media"].idxmax()]
    print(f"  {p} PCA, {c:<7}: max_depth={int(b['max_depth']):<3} min_samples_leaf={int(b['min_samples_leaf']):<3} "
          f"F1 {b['f1_media']:.4f} (+-{b['f1_desv']:.4f}) | recall {b['recall_media']:.4f} | "
          f"precision {b['precision_media']:.4f} | exactitud {b['exactitud_media']:.4f}")

b = tabla.loc[tabla["f1_media"].idxmax()]
print(f"\nMejor global: {b['pca']} PCA, {b['criterio']}, max_depth={int(b['max_depth'])}, "
      f"min_samples_leaf={int(b['min_samples_leaf'])}, F1 {b['f1_media']:.4f}")
print("\nLas 5 mejores configuraciones:")
print(tabla.nlargest(5, "f1_media")[["pca", "criterio", "max_depth", "min_samples_leaf",
                                     "f1_media", "f1_desv"]].round(4).to_string(index=False))
print("\nGuardado: resultados/fase5_arbol.csv")
