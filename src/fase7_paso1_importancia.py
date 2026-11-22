# Fase 7, Paso 1: importancia de caracteristicas del arbol final y arbol pequeno de ilustracion
# Usa SOLO entrenamiento (datos/train.csv). No abre la prueba.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from preprocesamiento import COLS, COLS_ORDEN_SALIDA, construir_pipeline
from validacion import crear_cv

tr = pd.read_csv("datos/train.csv")
X, y = tr[COLS], tr["y"].to_numpy()

def arbol_final():
    return DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=5, random_state=42)

# 1) Importancia en el arbol final (todo el entrenamiento). Ojo: el ColumnTransformer reordena las columnas
pipe = construir_pipeline(arbol_final()).fit(X, y)
imp = pd.Series(pipe.named_steps["modelo"].feature_importances_, index=COLS_ORDEN_SALIDA)
assert abs(imp.sum() - 1) < 1e-9

# 2) Estabilidad: mismo arbol en cada uno de los 5 pliegues de entrenamiento (mismos pliegues de la Fase 5)
filas = []
for itr, _ in crear_cv().split(X, y):
    p = construir_pipeline(arbol_final()).fit(X.iloc[itr], y[itr])
    filas.append(pd.Series(p.named_steps["modelo"].feature_importances_, index=COLS_ORDEN_SALIDA))
pl = pd.DataFrame(filas)

# 3) Comparacion con los coeficientes de la regresion logistica (Fase 5): solo el orden por tamano |coef|
coef = pd.read_csv("resultados/fase5_logreg_coeficientes.csv", index_col=0)["coeficiente"]
tabla = pd.DataFrame({"importancia_arbol": imp, "pliegues_media": pl.mean(), "pliegues_desv": pl.std(),
                      "pliegues_min": pl.min(), "pliegues_max": pl.max(),
                      "logreg_coef": coef.reindex(imp.index)})
tabla["rango_arbol"] = tabla["importancia_arbol"].rank(ascending=False).astype(int)
tabla["rango_logreg_abs"] = tabla["logreg_coef"].abs().rank(ascending=False).astype(int)
tabla = tabla.sort_values("importancia_arbol", ascending=False)
tabla.index.name = "caracteristica"
tabla.round(4).to_csv("resultados/fase7_importancia_arbol.csv")
pd.set_option("display.width", 200)
print("Importancia de caracteristicas del arbol final (entrenamiento) y estabilidad en 5 pliegues\n")
print(tabla.round(3).to_string())
print(f"\nSuma de importancias = {imp.sum():.3f} | profundidad real del arbol = {pipe.named_steps['modelo'].get_depth()} "
      f"| hojas = {pipe.named_steps['modelo'].get_n_leaves()}")

# 4) Figura de importancias (barras = arbol final; puntos = 5 pliegues)
orden = tabla.index[::-1]
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh(orden, tabla.loc[orden, "importancia_arbol"], color="#6baed6", label="Arbol final (todo el entrenamiento)")
for i, c in enumerate(orden):
    ax.plot(pl[c], [i] * len(pl), "o", color="black", ms=3.5, alpha=0.7, label="Cada uno de los 5 pliegues" if i == 0 else None)
ax.set_xlabel("Importancia (reduccion de Gini, suma = 1)"); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("figuras/fig_fase7_importancia_arbol.png", dpi=300); plt.close(fig)

# 5) Arbol pequeno SOLO de ilustracion (profundidad 3, caracteristicas originales: un arbol no necesita escalar ni log)
peq = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=20, random_state=42).fit(X, y)
fig, ax = plt.subplots(figsize=(13, 6.5))
plot_tree(peq, feature_names=COLS, class_names=["Sin RD", "Con RD"], filled=True, rounded=True, impurity=True, fontsize=8, ax=ax)
fig.tight_layout(); fig.savefig("figuras/fig_fase7_arbol_ilustrativo.png", dpi=300); plt.close(fig)
print(f"Arbol de ilustracion (profundidad 3, hojas >= 20 fotos): exactitud en entrenamiento = {peq.score(X, y):.3f} (no es el modelo final)")
print("\nGuardado: resultados/fase7_importancia_arbol.csv, figuras/fig_fase7_importancia_arbol.png y figuras/fig_fase7_arbol_ilustrativo.png")
