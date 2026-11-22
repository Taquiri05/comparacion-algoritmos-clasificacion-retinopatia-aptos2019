# src\fase4_paso2_auc.py
# FASE 4 - PASO 2: comparar las 9 caracteristicas entre sin RD y con RD. USA SOLO datos\train.csv
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(RAIZ, "datos", "train.csv"))      # el test NO se abre
COLS = ["area_rojas", "n_rojas", "area_exudados", "n_exudados", "densidad_vasos",
        "glcm_contraste", "glcm_homogeneidad", "glcm_energia", "glcm_correlacion"]
print(f"Fotos de entrenamiento: {len(tr)} | sin RD: {(tr['y']==0).sum()} | con RD: {(tr['y']==1).sum()}")

filas = []
for c in COLS:
    auc = roc_auc_score(tr["y"], tr[c])            # AUC univariado (empates = 0.5)
    filas.append({"caracteristica": c,
                  "mediana_sin_RD": tr.loc[tr["y"] == 0, c].median(),
                  "mediana_con_RD": tr.loc[tr["y"] == 1, c].median(),
                  "AUC": auc,
                  "sentido": "mayor en con RD" if auc >= 0.5 else "mayor en sin RD",
                  "fuerza": max(auc, 1 - auc)})
res = pd.DataFrame(filas).sort_values("fuerza", ascending=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
res.to_csv(os.path.join(RAIZ, "resultados", "auc_univariado_train.csv"), index=False)

# Graficos de cajas por clase (sin mostrar los puntos extremos para que se vea la caja)
fig, ax = plt.subplots(3, 3, figsize=(11, 9))
for a, c in zip(ax.ravel(), COLS):
    a.boxplot([tr.loc[tr["y"] == 0, c], tr.loc[tr["y"] == 1, c]], tick_labels=["Sin RD", "Con RD"], showfliers=False)
    a.set_title(f"{c} (AUC = {roc_auc_score(tr['y'], tr[c]):.3f})", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RAIZ, "figuras", "fig_fase4_caracteristicas_por_clase.png"), dpi=300)
print("\nGuardados: resultados\\auc_univariado_train.csv y figuras\\fig_fase4_caracteristicas_por_clase.png")
