# Fase 6, Paso 2: entrenamiento final con TODO el entrenamiento y evaluacion en la prueba (UNA SOLA VEZ)
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from preprocesamiento import COLS, construir_pipeline

SEGURO = "resultados/fase6_ejecutado.txt"
if os.path.exists(SEGURO):
    sys.exit("La prueba ya se evaluo una vez (existe resultados/fase6_ejecutado.txt). No se vuelve a ejecutar.")

# 0) Comprobar que las configuraciones coinciden con las elegidas en la Fase 5 (nada cambia despues de ver la prueba)
cfg = pd.read_csv("resultados/configuraciones_elegidas.csv").set_index("modelo")["hiperparametros"]
assert cfg["Regresion logistica"] == "C=1"
assert cfg["KNN"] == "K=11, pesos=uniform"
assert cfg["Arbol de decision"] == "criterio=gini, max_depth=8, min_samples_leaf=5"
T_LOGREG = float(pd.read_csv("resultados/umbral_logreg_elegido.csv")["umbral"].iloc[0])   # 0.37, fijado en la Fase 5

tr = pd.read_csv("datos/train.csv")
te = pd.read_csv("datos/test.csv")             # <- UNICA vez que se abre la prueba
assert set(tr["id_code"]).isdisjoint(set(te["id_code"]))
Xtr, ytr = tr[COLS], tr["y"].to_numpy()
Xte, yte = te[COLS], te["y"].to_numpy()

# 1) Entrenar con todo el entrenamiento (sin PCA) y obtener probabilidades en la prueba
modelos = {
    "Regresion logistica": LogisticRegression(C=1, max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=11, weights="uniform"),
    "Arbol de decision": DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=5, random_state=42),
}
proba = {}
for nombre, m in modelos.items():
    pipe = construir_pipeline(m).fit(Xtr, ytr)
    proba[nombre] = pipe.predict_proba(Xte)[:, 1]

# 2) Filas a evaluar: los 3 modelos con umbral 0.5 y la regresion logistica tambien con 0.37
filas = [("Regresion logistica", 0.5), ("Regresion logistica", T_LOGREG), ("KNN", 0.5), ("Arbol de decision", 0.5)]

def metricas(y, p, t):
    pred = (p >= t).astype(int)
    tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == 0) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0)); tn = np.sum((pred == 0) & (y == 0))
    rec = tp / (tp + fn); esp = tn / (tn + fp)
    pre = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else np.nan
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "exactitud": (tp + tn) / len(y),
            "precision": pre, "recall": rec, "especificidad": esp, "F1": f1, "AUC": roc_auc_score(y, p)}

resultados = []
for nombre, t in filas:
    r = {"modelo": nombre, "umbral": t}; r.update(metricas(yte, proba[nombre], t)); resultados.append(r)
res = pd.DataFrame(resultados)

# 3) Bootstrap (complemento): 1000 remuestreos de la prueba, mismos indices para todos los modelos
rng = np.random.default_rng(42)
B = 1000
idx = rng.integers(0, len(yte), size=(B, len(yte)))
boot = {(n, t): [] for n, t in filas}
for b in range(B):
    ib = idx[b]
    for n, t in filas:
        boot[(n, t)].append(metricas(yte[ib], proba[n][ib], t))
for col in ["exactitud", "precision", "recall", "especificidad", "F1", "AUC"]:
    lo, hi = [], []
    for n, t in filas:
        v = np.array([m[col] for m in boot[(n, t)]], dtype=float)
        lo.append(np.nanpercentile(v, 2.5)); hi.append(np.nanpercentile(v, 97.5))
    res[col + "_IC_inf"] = lo; res[col + "_IC_sup"] = hi

# Diferencias de F1 entre modelos (a 0.5) con su intervalo bootstrap
pares = [("KNN", "Regresion logistica"), ("KNN", "Arbol de decision"), ("Arbol de decision", "Regresion logistica")]
dif = []
for a, b_ in pares:
    d = np.array([boot[(a, 0.5)][i]["F1"] - boot[(b_, 0.5)][i]["F1"] for i in range(B)])
    dif.append({"comparacion": f"{a} - {b_}", "dif_F1": res[(res.modelo == a) & (res.umbral == 0.5)]["F1"].iloc[0]
                - res[(res.modelo == b_) & (res.umbral == 0.5)]["F1"].iloc[0],
                "IC_inf": np.percentile(d, 2.5), "IC_sup": np.percentile(d, 97.5)})
dif = pd.DataFrame(dif)

# 4) Comparar con la validacion cruzada de la Fase 5
cv = {"Regresion logistica": 0.8595, "KNN": 0.8842, "Arbol de decision": 0.8628}
res["F1_validacion_cruzada"] = res["modelo"].map(cv)

# 5) Imprimir
print(f"Prueba: {len(te)} fotos ({int(yte.sum())} con RD, {int((1-yte).sum())} sin RD)\n")
for _, r in res.iterrows():
    print(f"{r['modelo']} (umbral {r['umbral']:.2f})")
    print(f"  TP {int(r['TP'])} | FN {int(r['FN'])} | FP {int(r['FP'])} | TN {int(r['TN'])}")
    for c in ["exactitud", "precision", "recall", "especificidad", "F1", "AUC"]:
        print(f"  {c:<14}{r[c]:.4f}  [{r[c+'_IC_inf']:.4f}, {r[c+'_IC_sup']:.4f}]")
    print(f"  F1 en validacion cruzada (Fase 5): {r['F1_validacion_cruzada']:.4f}  ->  diferencia {r['F1'] - r['F1_validacion_cruzada']:+.4f}\n")
print("Diferencias de F1 entre modelos (umbral 0.5), con intervalo bootstrap de 95 %:")
print(dif.round(4).to_string(index=False))

# 6) Guardar todo y cerrar el seguro
res.to_csv("resultados/fase6_metricas_prueba.csv", index=False)
dif.to_csv("resultados/fase6_diferencias_bootstrap.csv", index=False)
pd.DataFrame({"id_code": te["id_code"], "y": yte, **{f"p_{n}": p for n, p in proba.items()}}
             ).to_csv("resultados/fase6_predicciones_prueba.csv", index=False)
with open(SEGURO, "w") as f:
    f.write("La prueba se evaluo una sola vez. No volver a ejecutar fase6_paso2_prueba.py.\n")
print("\nGuardado: resultados/fase6_metricas_prueba.csv, fase6_diferencias_bootstrap.csv, fase6_predicciones_prueba.csv")
