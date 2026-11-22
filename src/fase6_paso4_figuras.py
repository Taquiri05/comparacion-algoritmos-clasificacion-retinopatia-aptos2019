# Fase 6, Paso 4: figuras y tabla comparativa final (usa solo resultados ya guardados; no reentrena nada)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

pred = pd.read_csv("resultados/fase6_predicciones_prueba.csv")
met = pd.read_csv("resultados/fase6_metricas_prueba.csv")
cam = pd.read_csv("resultados/fase6_control_camara_prueba.csv")
base = pd.read_csv("resultados/fase6_linea_base_tamano.csv").iloc[0]
tr = pd.read_csv("datos/train.csv")[["id_code", "y"]]
eda = pd.read_csv("datos/eda_imagenes.csv")[["id_code", "tamano"]]
y = pred["y"].to_numpy()

# Linea base de tamano dentro del subconjunto: clase mayoritaria de 2588x1958 en entrenamiento
tr = tr.merge(eda, on="id_code", how="left")
may_sub = int(tr.loc[tr["tamano"] == "2588x1958", "y"].mean() >= 0.5)
es_sub = pred.merge(eda, on="id_code", how="left")["tamano"].eq("2588x1958").to_numpy()
base_sub = float(np.mean(y[es_sub] == may_sub))

CASOS = [("Regresion logistica", 0.5, "Regresion logistica (t = 0.50)"), ("Regresion logistica", 0.37, "Regresion logistica (t = 0.37)"),
         ("KNN", 0.5, "KNN (K = 11)"), ("Arbol de decision", 0.5, "Arbol de decision")]
COL = {"Regresion logistica": "#3182bd", "KNN": "#31a354", "Arbol de decision": "#de2d26"}

# ---------- Figura 1: matrices de confusion
fig, axs = plt.subplots(2, 2, figsize=(7.5, 6.5))
for ax, (m, t, titulo) in zip(axs.ravel(), CASOS):
    pr = (pred[f"p_{m}"].to_numpy() >= t).astype(int)
    cm = np.array([[np.sum((y == 0) & (pr == 0)), np.sum((y == 0) & (pr == 1))],
                   [np.sum((y == 1) & (pr == 0)), np.sum((y == 1) & (pr == 1))]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum() * 100:.1f} %)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Sin RD", "Con RD"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Sin RD", "Con RD"])
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Real"); ax.set_title(titulo, fontsize=10)
fig.tight_layout(); fig.savefig("figuras/fig_fase6_matrices_confusion.png", dpi=300); plt.close(fig)

# ---------- Figura 2: curvas ROC (3 modelos) con el punto de operacion de cada umbral
fig, ax = plt.subplots(figsize=(5.8, 5.2))
for m in ["Regresion logistica", "KNN", "Arbol de decision"]:
    p = pred[f"p_{m}"].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=COL[m], label=f"{m} (AUC = {roc_auc_score(y, p):.3f})")
    for t, marca in ([(0.5, "o"), (0.37, "s")] if m == "Regresion logistica" else [(0.5, "o")]):
        pr = (p >= t).astype(int)
        ax.plot(np.mean(pr[y == 0]), np.mean(pr[y == 1]), marca, color=COL[m], mec="black", ms=7)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.plot([], [], "o", color="gray", mec="black", label="Umbral 0.50"); ax.plot([], [], "s", color="gray", mec="black", label="Umbral 0.37 (reg. logistica)")
ax.set_xlabel("1 - Especificidad (falsos positivos)"); ax.set_ylabel("Recall (sensibilidad)"); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("figuras/fig_fase6_curvas_roc.png", dpi=300); plt.close(fig)

# ---------- Figura 3: exactitud en la prueba completa y en el subconjunto 2588x1958, con la linea base de tamano
modelos = ["Regresion logistica", "KNN", "Arbol de decision"]
completa = met[met["umbral"] == 0.5].set_index("modelo").loc[modelos]
sub = cam[cam["umbral"] == 0.5].set_index("modelo").loc[modelos]
x = np.arange(3); w = 0.36
fig, ax = plt.subplots(figsize=(7.2, 5.4))
for k, (datos, pref, desp, lbl) in enumerate([(completa, "exactitud", -w / 2, "Prueba completa (706)"),
                                              (sub, "exactitud_sub", w / 2, f"Subconjunto 2588x1958 ({int(es_sub.sum())})")]):
    v = datos[pref].to_numpy()
    lo = v - datos[f"{pref}_IC_inf"].to_numpy(); hi = datos[f"{pref}_IC_sup"].to_numpy() - v
    ax.bar(x + desp, v, w, yerr=[lo, hi], capsize=3, color=["#6baed6", "#fd8d3c"][k], label=lbl)
ax.hlines(float(base["exactitud"]), -0.5, 2.5, colors="#08519c", linestyles="--", lw=1.2, label=f"Linea base de tamano, prueba completa ({base['exactitud']:.3f})")
ax.hlines(base_sub, -0.5, 2.5, colors="#a63603", linestyles="--", lw=1.2, label=f"Linea base en el subconjunto ({base_sub:.3f})")
ax.set_xticks(x); ax.set_xticklabels(modelos); ax.set_ylim(0.4, 1.0); ax.set_ylabel("Exactitud (barras: IC 95 % bootstrap)")
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False); fig.tight_layout()
fig.savefig("figuras/fig_fase6_comparacion_camara.png", dpi=300); plt.close(fig)

# ---------- Tabla comparativa final
def fmt(v, lo, hi): return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
filas = []
for m, t, titulo in CASOS:
    a = met[(met["modelo"] == m) & (met["umbral"] == t)].iloc[0]
    c = cam[(cam["modelo"] == m) & (cam["umbral"] == t)].iloc[0]
    filas.append({"modelo": titulo, "exactitud": fmt(a["exactitud"], a["exactitud_IC_inf"], a["exactitud_IC_sup"]),
                  "precision": f"{a['precision']:.3f}", "recall": fmt(a["recall"], a["recall_IC_inf"], a["recall_IC_sup"]),
                  "especificidad": f"{a['especificidad']:.3f}", "F1": fmt(a["F1"], a["F1_IC_inf"], a["F1_IC_sup"]),
                  "AUC": fmt(a["AUC"], a["AUC_IC_inf"], a["AUC_IC_sup"]),
                  "exactitud_subconjunto": fmt(c["exactitud_sub"], c["exactitud_sub_IC_inf"], c["exactitud_sub_IC_sup"]),
                  "AUC_subconjunto": fmt(c["AUC_sub"], c["AUC_sub_IC_inf"], c["AUC_sub_IC_sup"])})
filas.append({"modelo": "Linea base de tamano", "exactitud": f"{base['exactitud']:.3f}", "precision": "", "recall": f"{base['recall']:.3f}",
              "especificidad": f"{base['especificidad']:.3f}", "F1": f"{base['F1']:.3f}", "AUC": "",
              "exactitud_subconjunto": f"{base_sub:.3f}", "AUC_subconjunto": ""})
tabla = pd.DataFrame(filas)
tabla.to_csv("resultados/fase6_tabla_comparativa.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("Tabla comparativa final (prueba, IC 95 % bootstrap)\n")
print(tabla.to_string(index=False))
print("\nGuardado: resultados/fase6_tabla_comparativa.csv y 3 figuras en figuras/")
