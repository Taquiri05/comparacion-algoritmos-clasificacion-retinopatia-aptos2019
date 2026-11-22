# src\fase3_paso3.py
# FASE 3 - PASO 3: disco optico + exudados (y lesiones rojas corregidas) en la muestra
import cv2, numpy as np, glob, os, sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retina_utils import leer, preparar, disco_optico, vasos, lesiones_rojas, exudados

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
archivos = sorted(glob.glob(os.path.join(RAIZ, "muestra", "**", "*.png"), recursive=True))
print("Fotos encontradas:", len(archivos))

elegidas = []
for nivel in range(5):
    for f in archivos:
        if os.path.basename(f).startswith(f"nivel{nivel}"):
            elegidas.append(f); break
if len(elegidas) < 5:
    elegidas = archivos[:5]

fig, ax = plt.subplots(5, 2, figsize=(7, 17))
for i, f in enumerate(elegidas):
    img = leer(f)
    cuad, mc, vc = preparar(img)
    d, centro = disco_optico(cuad, mc)
    vas = vasos(vc, mc)
    rojas, n_r = lesiones_rojas(vc, mc, d, vas)
    exu, n_e = exudados(vc, mc, d, vas)
    px_ret = int(mc.sum()); px_disco = int((d * mc).sum())
    area_r = rojas.sum() / (px_ret - px_disco) * 100
    area_e = exu.sum() / (px_ret - px_disco) * 100      # pixeles exudado / (retina - disco) * 100
    print(os.path.basename(f), "| rojas: n=%d area=%.2f%%" % (n_r, area_r),
          "| exudados: n=%d area=%.2f%%" % (n_e, area_e), "| px disco:", px_disco)
    base = cv2.cvtColor(cuad, cv2.COLOR_BGR2RGB)
    marcada = base.copy()
    marcada[rojas > 0] = (0, 255, 0)        # lesiones rojas: verde
    marcada[exu > 0] = (255, 255, 0)        # exudados: amarillo
    cv2.circle(marcada, centro, 40, (0, 0, 255), 2)   # disco optico: circulo azul
    ax[i, 0].imshow(base); ax[i, 0].set_title("Retina 512x512", fontsize=9)
    ax[i, 1].imshow(marcada)
    ax[i, 1].set_title(f"Rojas n={n_r} ({area_r:.2f}%)  Exudados n={n_e} ({area_e:.2f}%)", fontsize=8)
    ax[i, 0].axis("off"); ax[i, 1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(RAIZ, "figuras", "fig_fase3_paso3_exudados.png"), dpi=300)
print("Figura guardada en figuras\\fig_fase3_paso3_exudados.png")
