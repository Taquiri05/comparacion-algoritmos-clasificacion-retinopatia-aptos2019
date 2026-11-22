# src\fase4_paso6_escalado.py
# FASE 4 - PASO 6: StandardScaler (z-score) ajustado SOLO con entrenamiento. NO abre test.csv
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["area_rojas", "n_rojas", "area_exudados", "n_exudados", "densidad_vasos",
        "glcm_contraste", "glcm_homogeneidad", "glcm_energia", "glcm_correlacion"]

# 1) Comprobacion del ejemplo a mano: {2,4,6,8} -> mu = 5, sigma = 2.236, z = -1.342, -0.447, 0.447, 1.342
ej = StandardScaler().fit(np.array([[2], [4], [6], [8]]))
print("Ejemplo a mano -> media:", ej.mean_[0], "| sigma:", round(float(np.sqrt(ej.var_[0])), 3),
      "| z:", ej.transform(np.array([[2], [4], [6], [8]])).ravel().round(3).tolist(),
      "| z de x=7:", round(float(ej.transform([[7]])[0, 0]), 3))

# 2) Escalado de las 9 caracteristicas con los datos de entrenamiento
tr = pd.read_csv(os.path.join(RAIZ, "datos", "train.csv"))
X = tr[COLS]
sc = StandardScaler().fit(X)                      # aprende mu y sigma SOLO de train
Z = pd.DataFrame(sc.transform(X), columns=COLS)

par = pd.DataFrame({"caracteristica": COLS, "media_train": sc.mean_, "desv_train": np.sqrt(sc.var_),
                    "media_despues": Z.mean().values, "desv_despues": Z.std(ddof=0).values})
pd.set_option("display.width", 200)
print("\nParametros aprendidos con entrenamiento (" + str(len(tr)) + " fotos):")
print(par.round(4).to_string(index=False))
par.to_csv(os.path.join(RAIZ, "resultados", "escalado_parametros_train.csv"), index=False)
print("\nControl: despues de escalar, cada columna tiene media ~0 y desviacion ~1:",
      bool(np.allclose(Z.mean(), 0, atol=1e-9) and np.allclose(Z.std(ddof=0), 1, atol=1e-9)))
print("Guardado: resultados\\escalado_parametros_train.csv")
