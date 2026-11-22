# src\fase4_paso4_correlacion.py
# FASE 4 - PASO 4: correlacion de Spearman entre las 9 caracteristicas. USA SOLO datos\train.csv y NO usa la etiqueta y
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(RAIZ, "datos", "train.csv"))      # el test NO se abre
COLS = ["area_rojas", "n_rojas", "area_exudados", "n_exudados", "densidad_vasos",
        "glcm_contraste", "glcm_homogeneidad", "glcm_energia", "glcm_correlacion"]

corr = tr[COLS].corr(method="spearman")                          # rho de Spearman
pd.set_option("display.width", 220)
print(corr.round(2).to_string())

# Pares con |rho| >= 0.80
pares = []
for i in range(len(COLS)):
    for j in range(i + 1, len(COLS)):
        r = corr.iloc[i, j]
        if abs(r) >= 0.80:
            pares.append({"caracteristica_1": COLS[i], "caracteristica_2": COLS[j], "rho": r})
pares = pd.DataFrame(pares, columns=["caracteristica_1", "caracteristica_2", "rho"])
print("\nPares con |rho| >= 0.80:")
print(pares.round(3).to_string(index=False) if len(pares) else "ninguno")
pares.to_csv(os.path.join(RAIZ, "resultados", "correlaciones_train.csv"), index=False)

# Mapa de calor
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLS, rotation=45, ha="right")
ax.set_yticks(range(len(COLS))); ax.set_yticklabels(COLS)
for i in range(len(COLS)):
    for j in range(len(COLS)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, label="rho de Spearman"); plt.tight_layout()
plt.savefig(os.path.join(RAIZ, "figuras", "fig_fase4_correlacion.png"), dpi=300)
print("\nGuardados: resultados\\correlaciones_train.csv y figuras\\fig_fase4_correlacion.png")
