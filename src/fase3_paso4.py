# src\fase3_paso4.py
# FASE 3 - PASO 4: densidad de vasos en la muestra
import cv2, numpy as np, glob, os, sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retina_utils import leer, preparar, disco_optico, densidad_vasos

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
    cuad, mc, vc = preparar(leer(f))
    d, centro = disco_optico(cuad, mc)
    esq, dens = densidad_vasos(vc, mc, d)
    px_valida = int(((mc > 0) & (d == 0)).sum())
    print(os.path.basename(f), "| px esqueleto:", int(esq.sum()), "| px retina valida:", px_valida,
          f"| densidad_vasos: {dens:.2f}%")
    base = cv2.cvtColor(cuad, cv2.COLOR_BGR2RGB)
    marcada = base.copy()
    marcada[cv2.dilate(esq.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (0, 255, 255)   # vasos en cian
    cv2.circle(marcada, centro, 40, (0, 0, 255), 2)
    ax[i, 0].imshow(base); ax[i, 0].set_title("Retina 512x512", fontsize=9)
    ax[i, 1].imshow(marcada); ax[i, 1].set_title(f"Vasos (esqueleto): densidad = {dens:.2f}%", fontsize=9)
    ax[i, 0].axis("off"); ax[i, 1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(RAIZ, "figuras", "fig_fase3_paso4_vasos.png"), dpi=300)
print("Figura guardada en figuras\\fig_fase3_paso4_vasos.png")
