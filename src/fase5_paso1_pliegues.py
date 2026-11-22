# Fase 5, Paso 1: crear los 5 pliegues SOLO con entrenamiento y comprobarlos
import numpy as np
import pandas as pd
from validacion import crear_cv

tr = pd.read_csv("datos/train.csv")          # test.csv NO se abre
y = tr["y"].to_numpy()
X = np.zeros((len(tr), 1))                   # los pliegues solo dependen de y
cv = crear_cv()

print(f"Entrenamiento: {len(tr)} fotos, {y.mean()*100:.2f} % con RD\n")
vistas = []                                  # indices de validacion de todos los pliegues
filas = []
for k, (i_ent, i_val) in enumerate(cv.split(X, y), start=1):
    assert len(set(i_ent) & set(i_val)) == 0          # ningun solapamiento
    vistas.extend(i_val)
    print(f"Pliegue {k}: entrena {len(i_ent)} | valida {len(i_val)} "
          f"(sin RD {np.sum(y[i_val]==0)}, con RD {np.sum(y[i_val]==1)}, "
          f"{y[i_val].mean()*100:.2f} % con RD)")
    for i in i_val:
        filas.append((tr.loc[i, "id_code"], k))

# cada foto debe quedar en validacion exactamente 1 vez
assert sorted(vistas) == list(range(len(tr)))
print("\nControl: cada foto valida exactamente una vez y no hay solapamiento: True")

pd.DataFrame(filas, columns=["id_code", "pliegue"]).to_csv(
    "resultados/pliegues_cv.csv", index=False)
print("Guardado: resultados/pliegues_cv.csv")
