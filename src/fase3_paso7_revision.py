# src\fase3_paso7_revision.py
# FASE 3 - PASO 7: revision de calidad de la tabla de caracteristicas (NO usa la etiqueta y)
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(RAIZ, "datos", "caracteristicas.csv"))
COLS = ["area_rojas", "n_rojas", "area_exudados", "n_exudados", "densidad_vasos",
        "glcm_contraste", "glcm_homogeneidad", "glcm_energia", "glcm_correlacion"]
X = df[COLS]

# 1) Revision de calidad: faltantes, infinitos, constantes, valores extremos (regla IQR: fuera de Q1-1.5*IQR / Q3+1.5*IQR)
filas = []
for c in COLS:
    s = X[c]
    q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
    extremos = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    filas.append({"caracteristica": c, "faltantes": int(s.isna().sum()),
                  "infinitos": int(np.isinf(s).sum()), "valores_distintos": int(s.nunique()),
                  "min": s.min(), "mediana": s.median(), "max": s.max(),
                  "extremos_IQR": extremos, "pct_extremos": round(extremos / len(s) * 100, 1),
                  "pct_ceros": round((s == 0).mean() * 100, 1)})
rev = pd.DataFrame(filas)
pd.set_option("display.width", 200)
print("Filas:", len(df), "| id_code repetidos:", int(df["id_code"].duplicated().sum()))
print(rev.round(3).to_string(index=False))
rev.to_csv(os.path.join(RAIZ, "resultados", "revision_caracteristicas.csv"), index=False)

# 2) Histograma de cada caracteristica (sin separar por clase)
fig, ax = plt.subplots(3, 3, figsize=(11, 9))
for a, c in zip(ax.ravel(), COLS):
    a.hist(X[c], bins=40, color="#4C72B0", edgecolor="white")
    a.set_title(c, fontsize=10); a.set_xlabel("valor"); a.set_ylabel("n.° de imágenes")
plt.tight_layout()
plt.savefig(os.path.join(RAIZ, "figuras", "fig_fase3_distribuciones.png"), dpi=300)
print("\nFigura guardada en figuras\\fig_fase3_distribuciones.png")
print("Tabla guardada en resultados\\revision_caracteristicas.csv")
