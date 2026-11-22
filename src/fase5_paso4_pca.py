# Fase 5, Paso 4: mirar el PCA en entrenamiento (varianza explicada y k para 90 %). No se entrena ningun modelo.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from preprocesamiento import COLS, COLS_ORDEN_SALIDA, construir_preprocesamiento
from validacion import crear_cv

UMBRAL = 0.90                                  # fijado de antemano
tr = pd.read_csv("datos/train.csv")            # test.csv NO se abre
X = tr[COLS]
y = tr["y"].to_numpy()                         # solo para colorear el grafico 2D y crear los pliegues

# 1) Vista general: log + z-score + PCA con las 9 componentes, ajustado con todo el entrenamiento
prep = construir_preprocesamiento()
Z = prep.fit_transform(X)
pca = PCA(n_components=9, random_state=42).fit(Z)
ve = pca.explained_variance_ratio_
acum = np.cumsum(ve)
k = int(np.argmax(acum >= UMBRAL) + 1)         # primer k con varianza acumulada >= 90 %

print("Varianza explicada por componente (todo el entrenamiento)\n")
for j in range(9):
    print(f"CP{j+1}: {ve[j]*100:5.2f} %   acumulada {acum[j]*100:6.2f} %")
print(f"\nk minimo para >= {UMBRAL*100:.0f} %: {k} componentes ({acum[k-1]*100:.2f} %)")

# 2) k en cada pliegue (PCA ajustado solo con los 4 pliegues de entrenamiento de esa vuelta)
print("\nk por pliegue (PCA ajustado solo con el entrenamiento de cada vuelta):")
for i, (i_ent, _) in enumerate(crear_cv().split(X, y), start=1):
    p = construir_preprocesamiento()
    Zf = p.fit_transform(X.iloc[i_ent])
    pf = PCA(n_components=UMBRAL, svd_solver="full", random_state=42).fit(Zf)
    print(f"  Pliegue {i}: k = {pf.n_components_}")

# 3) Pesos de cada componente (que columnas pesan en cada CP)
cargas = pd.DataFrame(pca.components_[:3].T, index=COLS_ORDEN_SALIDA, columns=["CP1", "CP2", "CP3"])
print("\nPesos de las 3 primeras componentes:")
print(cargas.round(3))

pd.DataFrame({"componente": [f"CP{j+1}" for j in range(9)], "varianza": ve, "acumulada": acum}
             ).to_csv("resultados/pca_varianza_train.csv", index=False)
cargas.to_csv("resultados/pca_pesos_train.csv")

# 4) Figuras (300 DPI)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(range(1, 10), ve * 100, color="#9ecae1", label="Por componente")
ax.plot(range(1, 10), acum * 100, "o-", color="#08519c", label="Acumulada")
ax.axhline(UMBRAL * 100, color="red", ls="--", lw=1, label=f"Umbral {UMBRAL*100:.0f} %")
ax.set_xlabel("Componente principal"); ax.set_ylabel("Varianza explicada (%)")
ax.set_xticks(range(1, 10)); ax.legend(); fig.tight_layout()
fig.savefig("figuras/fig_fase5_pca_varianza.png", dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
P = pca.transform(Z)                           # coordenadas de cada foto en las componentes
for clase, color, nombre in [(0, "#3182bd", "Sin RD"), (1, "#de2d26", "Con RD")]:
    m = y == clase
    ax.scatter(P[m, 0], P[m, 1], s=6, alpha=0.4, color=color, label=nombre)
ax.set_xlabel(f"CP1 ({ve[0]*100:.1f} %)"); ax.set_ylabel(f"CP2 ({ve[1]*100:.1f} %)")
ax.legend(markerscale=3); fig.tight_layout()
fig.savefig("figuras/fig_fase5_pca_2d.png", dpi=300); plt.close(fig)
print("\nGuardado: resultados/pca_varianza_train.csv, resultados/pca_pesos_train.csv,"
      " figuras/fig_fase5_pca_varianza.png, figuras/fig_fase5_pca_2d.png")
