# src\fase4_paso3_camara.py
# FASE 4 - PASO 3: control de camara. Repite el AUC solo en las fotos de entrenamiento de tamano 2588x1958. USA SOLO train.csv
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(RAIZ, "datos", "train.csv"))                  # el test NO se abre
eda = pd.read_csv(os.path.join(RAIZ, "datos", "eda_imagenes.csv"))[["id_code", "tamano"]]
tr = tr.merge(eda, on="id_code", how="left")                                 # agrega el tamano solo a las fotos de train
assert tr["tamano"].notna().all(), "faltan tamanos"

sub = tr[tr["tamano"] == "2588x1958"]
print(f"Entrenamiento: {len(tr)} | subconjunto 2588x1958: {len(sub)} (sin RD: {(sub['y']==0).sum()}, con RD: {(sub['y']==1).sum()})")

COLS = ["area_rojas", "n_rojas", "area_exudados", "n_exudados", "densidad_vasos",
        "glcm_contraste", "glcm_homogeneidad", "glcm_energia", "glcm_correlacion"]
filas = []
for c in COLS:
    a_todo = roc_auc_score(tr["y"], tr[c])
    a_sub = roc_auc_score(sub["y"], sub[c])
    mismo_sentido = (a_todo >= 0.5) == (a_sub >= 0.5)
    fuerza_sub = max(a_sub, 1 - a_sub)
    pasa = bool(mismo_sentido and fuerza_sub >= 0.60)          # regla fijada antes de ver los resultados
    filas.append({"caracteristica": c, "AUC_todo_train": a_todo, "AUC_subconjunto": a_sub,
                  "mismo_sentido": mismo_sentido, "fuerza_subconjunto": fuerza_sub, "pasa_control": pasa})
res = pd.DataFrame(filas)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
print("\nPasan el control:", res.loc[res["pasa_control"], "caracteristica"].tolist())
res.to_csv(os.path.join(RAIZ, "resultados", "auc_control_camara_train.csv"), index=False)

# Grafico: AUC en todo el entrenamiento vs en el subconjunto de una sola camara
x = np.arange(len(COLS)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, res["AUC_todo_train"], w, label="Todo el entrenamiento", color="#4C72B0")
ax.bar(x + w/2, res["AUC_subconjunto"], w, label="Subconjunto 2588x1958", color="#DD8452")
ax.axhline(0.5, color="k", lw=1); ax.axhspan(0.4, 0.6, color="gray", alpha=0.15, label="Zona sin señal (0.4 a 0.6)")
ax.set_xticks(x); ax.set_xticklabels(COLS, rotation=35, ha="right"); ax.set_ylabel("AUC univariado"); ax.set_ylim(0, 1)
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(RAIZ, "figuras", "fig_fase4_control_camara.png"), dpi=300)
print("Guardados: resultados\\auc_control_camara_train.csv y figuras\\fig_fase4_control_camara.png")
