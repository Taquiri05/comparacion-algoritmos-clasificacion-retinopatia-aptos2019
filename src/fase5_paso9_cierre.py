# Fase 5, Paso 9: figuras de la fase y configuraciones finales de los 3 modelos (para la Fase 6)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from preprocesamiento import COLS, COLS_ORDEN_SALIDA, construir_pipeline

# ---------- 1) Cargar resultados de los pasos 3, 5, 6 y 7 (todo viene de validacion cruzada de entrenamiento)
lr_sin = pd.read_csv("resultados/fase5_logreg_sin_pca.csv"); lr_sin["pca"] = "sin"
lr_con = pd.read_csv("resultados/fase5_logreg_con_pca.csv"); lr_con["pca"] = "con"
lr = pd.concat([lr_sin, lr_con], ignore_index=True)
knn = pd.read_csv("resultados/fase5_knn.csv")
arb = pd.read_csv("resultados/fase5_arbol.csv")
umbral = float(pd.read_csv("resultados/umbral_logreg_elegido.csv")["umbral"].iloc[0])

# ---------- 2) Configuracion elegida por modelo: mayor F1 medio (misma regla para los tres)
def mejor(df):
    return df.loc[df["f1_media"].idxmax()]
b_lr, b_knn, b_arb = mejor(lr), mejor(knn), mejor(arb)
elegidas = pd.DataFrame([
    {"modelo": "Regresion logistica", "pca": b_lr["pca"], "hiperparametros": f"C={b_lr['C']:g}",
     "umbral_adicional": umbral, **{k: b_lr[k] for k in ["f1_media", "f1_desv", "recall_media", "precision_media", "exactitud_media"]}},
    {"modelo": "KNN", "pca": b_knn["pca"], "hiperparametros": f"K={int(b_knn['K'])}, pesos={b_knn['pesos']}",
     "umbral_adicional": np.nan, **{k: b_knn[k] for k in ["f1_media", "f1_desv", "recall_media", "precision_media", "exactitud_media"]}},
    {"modelo": "Arbol de decision", "pca": b_arb["pca"],
     "hiperparametros": f"criterio={b_arb['criterio']}, max_depth={int(b_arb['max_depth'])}, min_samples_leaf={int(b_arb['min_samples_leaf'])}",
     "umbral_adicional": np.nan, **{k: b_arb[k] for k in ["f1_media", "f1_desv", "recall_media", "precision_media", "exactitud_media"]}},
])
elegidas.to_csv("resultados/configuraciones_elegidas.csv", index=False)
print("Configuraciones finales (validacion cruzada de entrenamiento, umbral 0.5 salvo la fila adicional)\n")
print(elegidas.round(4).to_string(index=False))

# ---------- 3) Figura: F1 frente a K (KNN)
fig, ax = plt.subplots(figsize=(6.5, 4))
estilos = {("sin", "uniform"): ("#08519c", "-", "Sin PCA, uniform"), ("sin", "distance"): ("#6baed6", "--", "Sin PCA, distance"),
           ("con", "uniform"): ("#a63603", "-", "Con PCA, uniform"), ("con", "distance"): ("#fd8d3c", "--", "Con PCA, distance")}
for (p, w), g in knn.groupby(["pca", "pesos"]):
    c, ls, nombre = estilos[(p, w)]
    ax.plot(g["K"], g["f1_media"], marker="o", ms=3, color=c, ls=ls, label=nombre)
ax.set_xscale("log"); ax.set_xticks(sorted(knn["K"].unique())); ax.set_xticklabels(sorted(knn["K"].unique()))
ax.set_xlabel("K (vecinos)"); ax.set_ylabel("F1 medio (validacion cruzada)"); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("figuras/fig_fase5_knn_f1_vs_k.png", dpi=300); plt.close(fig)

# ---------- 4) Figura: F1 frente a profundidad (arbol), mejor F1 de cada profundidad
fig, ax = plt.subplots(figsize=(6.5, 4))
for p, color, nombre in [("sin", "#08519c", "Sin PCA"), ("con", "#a63603", "Con PCA")]:
    g = arb[arb["pca"] == p].groupby("max_depth")["f1_media"].max()
    ax.plot(g.index, g.values, marker="o", ms=4, color=color, label=nombre)
ax.set_xlabel("max_depth"); ax.set_ylabel("Mejor F1 medio (validacion cruzada)"); ax.legend(); fig.tight_layout()
fig.savefig("figuras/fig_fase5_arbol_f1_vs_profundidad.png", dpi=300); plt.close(fig)

# ---------- 5) Coeficientes de la regresion logistica (C elegido, sin PCA), ajustada con todo el entrenamiento
tr = pd.read_csv("datos/train.csv")            # test.csv NO se abre
pipe = construir_pipeline(LogisticRegression(C=float(b_lr["C"]), max_iter=1000))
pipe.fit(tr[COLS], tr["y"])
coef = pd.Series(pipe.named_steps["modelo"].coef_[0], index=COLS_ORDEN_SALIDA).sort_values()
print("\nCoeficientes de la regresion logistica (variables en log y z-score; positivo = sube la probabilidad de RD):")
print(coef.round(3).to_string())
coef.to_csv("resultados/fase5_logreg_coeficientes.csv", header=["coeficiente"])

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh(coef.index, coef.values, color=["#de2d26" if v > 0 else "#3182bd" for v in coef.values])
ax.axvline(0, color="black", lw=0.8); ax.set_xlabel("Coeficiente (variables estandarizadas)"); fig.tight_layout()
fig.savefig("figuras/fig_fase5_logreg_coeficientes.png", dpi=300); plt.close(fig)
print("\nGuardado: resultados/configuraciones_elegidas.csv, resultados/fase5_logreg_coeficientes.csv y 3 figuras en figuras/")
