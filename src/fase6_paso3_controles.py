# Fase 6, Paso 3: control de camara (subconjunto 2588x1958) y linea base de tamano, en la prueba
# NO reentrena nada: usa las predicciones ya guardadas en el Paso 2.
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

tr = pd.read_csv("datos/train.csv")[["id_code", "y"]]
pred = pd.read_csv("resultados/fase6_predicciones_prueba.csv")        # id_code, y, p_<modelo>
eda = pd.read_csv("datos/eda_imagenes.csv")[["id_code", "tamano"]]
tr = tr.merge(eda, on="id_code", how="left"); pred = pred.merge(eda, on="id_code", how="left")
assert tr["tamano"].notna().all() and pred["tamano"].notna().all(), "faltan tamanos"

def metricas(y, p, t):
    pr = (p >= t).astype(int)
    tp = np.sum((pr == 1) & (y == 1)); fn = np.sum((pr == 0) & (y == 1))
    fp = np.sum((pr == 1) & (y == 0)); tn = np.sum((pr == 0) & (y == 0))
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    esp = tn / (tn + fp) if (tn + fp) else np.nan
    pre = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else np.nan
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return {"exactitud": (tp + tn) / len(y), "recall": rec, "especificidad": esp, "F1": f1, "AUC": auc}

# ---------- 1) Linea base de tamano: clase mayoritaria de cada tamano, aprendida SOLO con entrenamiento
mayoritaria = tr.groupby("tamano")["y"].agg(lambda s: int(s.mean() >= 0.5))
global_may = int(tr["y"].mean() >= 0.5)
pred["base"] = pred["tamano"].map(mayoritaria).fillna(global_may).astype(int)
n_nuevos = int((~pred["tamano"].isin(mayoritaria.index)).sum())
b = metricas(pred["y"].to_numpy(), pred["base"].to_numpy().astype(float), 0.5)
print("LINEA BASE DE TAMANO en la prueba (regla aprendida con entrenamiento)")
print(f"  fotos de prueba con un tamano que no estaba en entrenamiento: {n_nuevos}")
print(f"  exactitud {b['exactitud']:.4f} | recall {b['recall']:.4f} | especificidad {b['especificidad']:.4f} | F1 {b['F1']:.4f}")
print("  (en validacion cruzada la exactitud fue 0.9034)\n")
pd.DataFrame([{"regla": "clase mayoritaria por tamano", **b}]).to_csv("resultados/fase6_linea_base_tamano.csv", index=False)

# ---------- 2) Control de camara: modelos en el subconjunto 2588x1958 frente al resto
modelos = [("Regresion logistica", "p_Regresion logistica", 0.5), ("Regresion logistica", "p_Regresion logistica", 0.37),
           ("KNN", "p_KNN", 0.5), ("Arbol de decision", "p_Arbol de decision", 0.5)]
es_sub = (pred["tamano"] == "2588x1958").to_numpy()
y = pred["y"].to_numpy()
print(f"CONTROL DE CAMARA. Prueba: {len(pred)} fotos | subconjunto 2588x1958: {es_sub.sum()} "
      f"(sin RD {int(((y == 0) & es_sub).sum())}, con RD {int(((y == 1) & es_sub).sum())}) | resto: {(~es_sub).sum()}")
sub_b = metricas(y[es_sub], pred["base"].to_numpy()[es_sub].astype(float), 0.5)
print(f"  Linea base de tamano en el subconjunto: exactitud {sub_b['exactitud']:.4f} (el tamano ya no separa las clases)\n")

rng = np.random.default_rng(42); B = 1000
idx_sub = np.flatnonzero(es_sub)
boot_idx = rng.integers(0, len(idx_sub), size=(B, len(idx_sub)))
filas = []
for nombre, col, t in modelos:
    p = pred[col].to_numpy()
    m_sub = metricas(y[es_sub], p[es_sub], t); m_rest = metricas(y[~es_sub], p[~es_sub], t)
    bt = [metricas(y[idx_sub[ib]], p[idx_sub[ib]], t) for ib in boot_idx]
    fila = {"modelo": nombre, "umbral": t, "n_sub": int(es_sub.sum())}
    for c in ["exactitud", "recall", "especificidad", "F1", "AUC"]:
        v = np.array([x[c] for x in bt], dtype=float)
        fila[f"{c}_sub"] = m_sub[c]; fila[f"{c}_sub_IC_inf"] = np.nanpercentile(v, 2.5); fila[f"{c}_sub_IC_sup"] = np.nanpercentile(v, 97.5)
        fila[f"{c}_resto"] = m_rest[c]
    filas.append(fila)
res = pd.DataFrame(filas)
res.to_csv("resultados/fase6_control_camara_prueba.csv", index=False)
for _, r in res.iterrows():
    print(f"{r['modelo']} (umbral {r['umbral']:.2f})")
    for c in ["exactitud", "recall", "especificidad", "F1", "AUC"]:
        print(f"  {c:<14} subconjunto {r[c+'_sub']:.4f} [{r[c+'_sub_IC_inf']:.4f}, {r[c+'_sub_IC_sup']:.4f}]   resto {r[c+'_resto']:.4f}")
print("\nGuardado: resultados/fase6_linea_base_tamano.csv, resultados/fase6_control_camara_prueba.csv")
