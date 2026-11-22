# src\fase3_paso2.py
# FASE 3 - PASO 2: lesiones rojas (area y cantidad) en la muestra
import cv2, numpy as np, glob, os, sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retina_utils import leer, preparar, lesiones_rojas

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
    rojas, n = lesiones_rojas(vc, mc)
    px_les, px_ret = int(rojas.sum()), int(mc.sum())
    area = px_les / px_ret * 100                 # area_rojas = pixeles lesion / pixeles retina * 100
    print(os.path.basename(f), "| n_rojas:", n, "| px lesion:", px_les,
          "| px retina:", px_ret, f"| area_rojas: {area:.2f}%")
    marcada = cv2.cvtColor(cuad, cv2.COLOR_BGR2RGB).copy()
    marcada[rojas > 0] = (0, 255, 0)             # lesiones detectadas en verde
    ax[i, 0].imshow(cv2.cvtColor(cuad, cv2.COLOR_BGR2RGB)); ax[i, 0].set_title("Retina 512x512", fontsize=9)
    ax[i, 1].imshow(marcada); ax[i, 1].set_title(f"Lesiones rojas: n={n}, area={area:.2f}%", fontsize=9)
    ax[i, 0].axis("off"); ax[i, 1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(RAIZ, "figuras", "fig_fase3_paso2_rojas.png"), dpi=300)
print("Figura guardada en figuras\\fig_fase3_paso2_rojas.png")
