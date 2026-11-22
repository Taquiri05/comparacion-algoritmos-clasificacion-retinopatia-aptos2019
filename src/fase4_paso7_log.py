# src\fase4_paso7_log.py
# FASE 4 - PASO 7: log(1+x) en 4 columnas + StandardScaler, ajustado SOLO con entrenamiento. NO abre test.csv
import os, sys, numpy as np, pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocesamiento import COLS, COLS_LOG, COLS_ORDEN_SALIDA, construir_preprocesamiento

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1) Comprobacion de los ejemplos a mano
print("log(1+x) area_rojas {0,0.05,0.15,3.5}:", np.log1p([0, 0.05, 0.15, 3.5]).round(3).tolist())
print("log(1+x) n_rojas {0,5,10,180}:", np.log1p([0, 5, 10, 180]).round(3).tolist())

# 2) Efecto en las 4 columnas, con datos de entrenamiento
tr = pd.read_csv(os.path.join(RAIZ, "datos", "train.csv"))
X = tr[COLS]
prep = construir_preprocesamiento().fit(X)                       # aprende mu y sigma SOLO de train
Z = pd.DataFrame(prep.transform(X), columns=COLS_ORDEN_SALIDA)   # columnas escaladas
Z_sin_log = pd.DataFrame(StandardScaler().fit_transform(X), columns=COLS)   # escalado sin logaritmo, para comparar

filas = []
for c in COLS_LOG:
    filas.append({"caracteristica": c,
                  "asimetria_sin_log": skew(X[c]), "asimetria_con_log": skew(np.log1p(X[c])),
                  "z_maximo_sin_log": Z_sin_log[c].max(), "z_maximo_con_log": Z[c].max()})
res = pd.DataFrame(filas)
pd.set_option("display.width", 200)
print("\nEfecto del logaritmo (solo entrenamiento, " + str(len(tr)) + " fotos):")
print(res.round(3).to_string(index=False))
res.to_csv(os.path.join(RAIZ, "resultados", "efecto_log_train.csv"), index=False)

ok = bool(np.allclose(Z.mean(), 0, atol=1e-9) and np.allclose(Z.std(ddof=0), 1, atol=1e-9))
print("\nControl: tras log + escalado, las 9 columnas tienen media ~0 y desviacion ~1:", ok)
print("Guardado: resultados\\efecto_log_train.csv")
