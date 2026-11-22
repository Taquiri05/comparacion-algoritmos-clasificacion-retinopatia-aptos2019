# Fase 7, Paso 3: panel de ejemplos de errores con imagenes reales (3 filas x 3 fotos)
# Seleccion fijada antes de ver las fotos: al azar con semilla 42 dentro de cada grupo. No cambia ningun modelo.
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARPETA_IMAGENES = Path("..") / "Dataset aptos"          # las imagenes NO se copian al proyecto
assert CARPETA_IMAGENES.exists(), f"No encuentro {CARPETA_IMAGENES.resolve()}"

c = pd.read_csv("resultados/fase7_errores_compartidos.csv")
GRUPOS = [("Falsos negativos de los 3 modelos, nivel 1", c[(c["tipo"] == "FN_los_3") & (c["diagnosis"] == 1)]),
          ("Falsos negativos de los 3 modelos, nivel 2", c[(c["tipo"] == "FN_los_3") & (c["diagnosis"] == 2)]),
          ("Falsos positivos de los 3 modelos, tamano 819x614 (sin RD)", c[(c["tipo"] == "FP_los_3") & (c["tamano"] == "819x614")])]

# Indice de imagenes por id_code (busca en todas las subcarpetas, no depende de como esten organizadas)
indice = {p.stem: p for p in CARPETA_IMAGENES.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")}

rng = np.random.default_rng(42)
elegidas = []
for nombre, g in GRUPOS:
    ids = sorted(g["id_code"])
    sel = [str(x) for x in rng.choice(ids, size=3, replace=False)]
    print(f"{nombre}: {len(ids)} casos disponibles -> elegidas {sel}")
    elegidas += [(nombre, i) for i in sel]
faltan = [i for _, i in elegidas if i not in indice]
assert not faltan, f"No encuentro estas imagenes en {CARPETA_IMAGENES.resolve()}: {faltan}"

def leer(ruta, lado=512):
    im = cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_COLOR)   # lectura segura con espacios y puntos en la ruta
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    f = lado / max(im.shape[:2])
    return cv2.resize(im, (int(im.shape[1] * f), int(im.shape[0] * f)), interpolation=cv2.INTER_AREA)

fig, axs = plt.subplots(3, 3, figsize=(10, 10.6))
filas = []
for ax, (grupo, i) in zip(axs.ravel(), elegidas):
    r = c[c["id_code"] == i].iloc[0]
    ax.imshow(leer(indice[i])); ax.axis("off")
    real = "sin RD (nivel 0)" if r["y"] == 0 else f"con RD (nivel {int(r['diagnosis'])})"
    ax.set_title(f"{i} | real: {real}\n{r['tamano']} | prob. con RD: LR {r['p_Regresion logistica']:.2f}, KNN {r['p_KNN']:.2f}, Arbol {r['p_Arbol de decision']:.2f}", fontsize=7.5)
    filas.append({"grupo": grupo, **r.to_dict()})
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.subplots_adjust(hspace=0.62)
for k, (nombre, _) in enumerate(GRUPOS):      # titulo del grupo justo encima de su fila
    fig.text(0.5, axs[k, 0].get_position().y1 + 0.043, nombre, ha="center", fontsize=10, fontweight="bold")
fig.savefig("figuras/fig_fase7_ejemplos_errores.png", dpi=300); plt.close(fig)
pd.DataFrame(filas).to_csv("resultados/fase7_ejemplos_elegidos.csv", index=False)
print("\nGuardado: figuras/fig_fase7_ejemplos_errores.png (contiene imagenes del dataset: NO subir a GitHub) y resultados/fase7_ejemplos_elegidos.csv")
