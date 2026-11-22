# Fase 7, Paso 2: analisis de errores en la prueba (solo descripcion; no se cambia ningun modelo)
# Usa las predicciones ya guardadas en la Fase 6: NO reentrena ni vuelve a abrir test.csv.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pred = pd.read_csv("resultados/fase6_predicciones_prueba.csv")
eda = pd.read_csv("datos/eda_imagenes.csv")[["id_code", "tamano"]]
car = pd.read_csv("datos/caracteristicas.csv")[["id_code", "diagnosis"]]   # nivel 0-4 de cada foto
d = pred.merge(car, on="id_code", how="left").merge(eda, on="id_code", how="left")
assert len(d) == len(pred) == 706 and d["diagnosis"].notna().all() and d["tamano"].notna().all()
assert ((d["diagnosis"] > 0).astype(int) == d["y"]).all(), "El nivel no coincide con y"

CASOS = {"LogReg t=0.50": ("Regresion logistica", 0.5), "LogReg t=0.37": ("Regresion logistica", 0.37),
         "KNN": ("KNN", 0.5), "Arbol": ("Arbol de decision", 0.5)}
for k, (m, t) in CASOS.items():
    d["pred_" + k] = (d["p_" + m] >= t).astype(int)

def wilson(x, n, z=1.96):
    p = x / n; c = 1 + z**2 / n
    return (p + z**2 / (2 * n)) / c, z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / c   # centro, mitad del intervalo

# ---------- 1) Por nivel: nivel 0 = especificidad (acierta "sin RD"); niveles 1-4 = recall (acierta "con RD")
filas = []
for nivel in range(5):
    s = d[d["diagnosis"] == nivel]
    for k in CASOS:
        ok = int((s["pred_" + k] == (0 if nivel == 0 else 1)).sum()); n = len(s)
        c, h = wilson(ok, n)
        filas.append({"nivel": nivel, "modelo": k, "n": n, "aciertos": ok, "tasa": ok / n, "IC_inf": max(c - h, 0), "IC_sup": min(c + h, 1)})
nivel = pd.DataFrame(filas)
nivel.round(4).to_csv("resultados/fase7_errores_por_nivel.csv", index=False)
ver = nivel.assign(v=lambda t: t["aciertos"].astype(str) + "/" + t["n"].astype(str) + " = " + t["tasa"].round(3).astype(str)) \
           .pivot(index="nivel", columns="modelo", values="v")[list(CASOS)]
print("1) Aciertos por nivel en la prueba (nivel 0 = especificidad; niveles 1 a 4 = recall)\n")
print(ver.to_string())

# ---------- 2) Por tamano de imagen (camara): falsos positivos y falsos negativos
filas = []
for tam, s in d.groupby("tamano"):
    f = {"tamano": tam, "n_sinRD": int((s["y"] == 0).sum()), "n_conRD": int((s["y"] == 1).sum())}
    for k in CASOS:
        f["FP_" + k] = int(((s["y"] == 0) & (s["pred_" + k] == 1)).sum())
        f["FN_" + k] = int(((s["y"] == 1) & (s["pred_" + k] == 0)).sum())
    filas.append(f)
tam = pd.DataFrame(filas).sort_values(["n_sinRD", "n_conRD"], ascending=False)
tam.to_csv("resultados/fase7_errores_por_tamano.csv", index=False)
cols = ["tamano", "n_sinRD", "n_conRD"] + [f"FP_{k}" for k in ["LogReg t=0.50", "KNN", "Arbol"]] + [f"FN_{k}" for k in ["LogReg t=0.50", "KNN", "Arbol"]]
print("\n2) Errores por tamano de imagen (FP = sin RD marcada con RD; FN = con RD marcada sin RD; umbral 0.5)\n")
print(tam[cols].rename(columns=lambda c: c.replace("LogReg t=0.50", "LR")).to_string(index=False))

# ---------- 3) Errores compartidos por los 3 modelos (umbral 0.5)
tres = ["LogReg t=0.50", "KNN", "Arbol"]
fn3 = d[(d["y"] == 1) & (d[["pred_" + k for k in tres]].sum(axis=1) == 0)].assign(tipo="FN_los_3")
fp3 = d[(d["y"] == 0) & (d[["pred_" + k for k in tres]].sum(axis=1) == 3)].assign(tipo="FP_los_3")
comp = pd.concat([fn3, fp3])[["id_code", "tipo", "y", "diagnosis", "tamano", "p_Regresion logistica", "p_KNN", "p_Arbol de decision"]]
comp.round(3).to_csv("resultados/fase7_errores_compartidos.csv", index=False)
fn_cada = {k: int(((d["y"] == 1) & (d["pred_" + k] == 0)).sum()) for k in tres}
fn_alguno = int(((d["y"] == 1) & (d[["pred_" + k for k in tres]].sum(axis=1) < 3)).sum())
print(f"\n3) Falsos negativos (umbral 0.5): {fn_cada}")
print(f"   Con RD que fallo al menos 1 modelo: {fn_alguno} | que fallaron los 3: {len(fn3)}")
print("   Falsos negativos de los 3 modelos por nivel:", fn3["diagnosis"].value_counts().sort_index().to_dict())
print("   Falsos negativos de los 3 modelos por tamano:", fn3["tamano"].value_counts().to_dict())
print(f"   Falsos positivos de los 3 modelos (sin RD marcadas con RD): {len(fp3)}; por tamano:", fp3["tamano"].value_counts().to_dict())

# ---------- Figura: acierto por nivel con IC 95 % (Wilson)
col = {"LogReg t=0.50": "#3182bd", "LogReg t=0.37": "#9ecae1", "KNN": "#31a354", "Arbol": "#de2d26"}
fig, ax = plt.subplots(figsize=(8, 4.8)); w = 0.2
for j, k in enumerate(CASOS):
    s = nivel[nivel["modelo"] == k].sort_values("nivel")
    ax.bar(s["nivel"] + (j - 1.5) * w, s["tasa"], w, color=col[k], label=k,
           yerr=[s["tasa"] - s["IC_inf"], s["IC_sup"] - s["tasa"]], capsize=2, error_kw={"lw": 0.8})
n_niv = nivel[nivel["modelo"] == "KNN"].set_index("nivel")["n"]
ax.set_xticks(range(5)); ax.set_xticklabels([f"Nivel {i}\n(n = {n_niv[i]})" + ("\nespecificidad" if i == 0 else "\nrecall") for i in range(5)])
ax.set_ylim(0, 1.05); ax.set_ylabel("Tasa de acierto (barras: IC 95 % de Wilson)"); ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.tight_layout(); fig.savefig("figuras/fig_fase7_acierto_por_nivel.png", dpi=300); plt.close(fig)
print("\nGuardado: resultados/fase7_errores_por_nivel.csv, fase7_errores_por_tamano.csv, fase7_errores_compartidos.csv y figuras/fig_fase7_acierto_por_nivel.png")
