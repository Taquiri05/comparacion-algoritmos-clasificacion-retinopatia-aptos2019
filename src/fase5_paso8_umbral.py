# Fase 5, Paso 8: umbral de la regresion logistica (C=1, sin PCA) con probabilidades de validacion cruzada
# Regla fijada de antemano: el umbral MAS ALTO con recall >= 0.90
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from preprocesamiento import COLS, construir_pipeline
from validacion import crear_cv

RECALL_MIN = 0.90
tr = pd.read_csv("datos/train.csv")            # test.csv NO se abre
X = tr[COLS]
y = tr["y"].to_numpy()

# Probabilidad de cada foto calculada por un modelo que NO la vio (validacion cruzada, mismos pliegues)
pipe = construir_pipeline(LogisticRegression(C=1, max_iter=1000))
p = cross_val_predict(pipe, X, y, cv=crear_cv(), method="predict_proba")[:, 1]

filas = []
for t in np.round(np.arange(0.10, 0.901, 0.01), 2):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rec = tp / (tp + fn); esp = tn / (tn + fp); pre = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else np.nan
    filas.append({"umbral": t, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                  "recall": rec, "especificidad": esp, "precision": pre, "F1": f1})
tabla = pd.DataFrame(filas)
tabla.to_csv("resultados/fase5_umbral_logreg.csv", index=False)

print("Umbral de la regresion logistica (probabilidades de validacion cruzada, solo entrenamiento)\n")
print(tabla[tabla["umbral"].round(2).isin([0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90])]
      .round(4).to_string(index=False))

cumple = tabla[tabla["recall"] >= RECALL_MIN]
if len(cumple) == 0:
    print(f"\nNingun umbral entre 0.10 y 0.90 logra recall >= {RECALL_MIN}")
else:
    e = cumple.loc[cumple["umbral"].idxmax()]            # el mas alto que cumple la regla
    b = tabla[tabla["umbral"] == 0.5].iloc[0]
    print(f"\nRegla: umbral mas alto con recall >= {RECALL_MIN}  ->  t = {e['umbral']:.2f}")
    print(f"  con t=0.50: recall {b['recall']:.4f} | especificidad {b['especificidad']:.4f} | "
          f"precision {b['precision']:.4f} | F1 {b['F1']:.4f} | FN {int(b['FN'])} | FP {int(b['FP'])}")
    print(f"  con t={e['umbral']:.2f}: recall {e['recall']:.4f} | especificidad {e['especificidad']:.4f} | "
          f"precision {e['precision']:.4f} | F1 {e['F1']:.4f} | FN {int(e['FN'])} | FP {int(e['FP'])}")
    pd.DataFrame({"modelo": ["regresion_logistica_C1_sin_PCA"], "umbral": [e["umbral"]]}
                 ).to_csv("resultados/umbral_logreg_elegido.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 4))
for col, color in [("recall", "#de2d26"), ("especificidad", "#3182bd"), ("precision", "#31a354"), ("F1", "#756bb1")]:
    ax.plot(tabla["umbral"], tabla[col], label=col.capitalize(), color=color)
ax.axhline(RECALL_MIN, color="gray", ls=":", lw=1)
ax.axvline(0.5, color="black", ls="--", lw=1, label="Umbral 0.5")
if len(cumple):
    ax.axvline(e["umbral"], color="red", ls="--", lw=1, label=f"Umbral elegido {e['umbral']:.2f}")
ax.set_xlabel("Umbral t"); ax.set_ylabel("Valor"); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("figuras/fig_fase5_umbral_logreg.png", dpi=300); plt.close(fig)
print("\nGuardado: resultados/fase5_umbral_logreg.csv, resultados/umbral_logreg_elegido.csv, figuras/fig_fase5_umbral_logreg.png")
