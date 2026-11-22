# src\fase3_paso1.py
# FASE 3 - PASO 1: encontrar la retina, recortarla a 512x512, quitar el borde y mejorar el contraste
import cv2, numpy as np, glob, os
import matplotlib.pyplot as plt

RAIZ = r"C:\Users\Jhonatan\Documents\IA ..... 1\PROYECTO INTEGRADOR\IA1_Proyecto"
CARPETA = os.path.join(RAIZ, "muestra")   # tus 25 fotos de prueba
LADO = 512

def leer(ruta):
    # cv2.imread falla con rutas con caracteres raros, por eso se lee asi
    return cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)

def preparar(img):
    # 1) ENCONTRAR LA RETINA: gris + suavizado; pixel > 10 = retina (el fondo negro vale ~0)
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    m = (g > 10).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])        # region conectada mas grande
    m = (lab == k).astype(np.uint8)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m); cv2.drawContours(m, cnt, -1, 1, -1)   # rellena huecos

    # 2) RECORTAR, CUADRAR (rellena con negro, no estira) y llevar a 512x512
    x, y, w, h = cv2.boundingRect(m)
    img, m = img[y:y+h, x:x+w], m[y:y+h, x:x+w]
    L = max(w, h); top, left = (L-h)//2, (L-w)//2
    cuad = np.zeros((L, L, 3), np.uint8); mc = np.zeros((L, L), np.uint8)
    cuad[top:top+h, left:left+w] = img; mc[top:top+h, left:left+w] = m
    cuad = cv2.resize(cuad, (LADO, LADO), interpolation=cv2.INTER_AREA)
    mc = cv2.resize(mc, (LADO, LADO), interpolation=cv2.INTER_NEAREST)

    # 3) QUITAR EL BORDE: encoge la mascara 15 px hacia adentro
    mc = cv2.erode(mc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cuad = cuad * mc[:, :, None]

    # 4) CONTRASTE: canal verde + CLAHE (bloques de 8x8, clipLimit 2.0)
    verde = cuad[:, :, 1]
    vc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(verde) * mc
    return cuad, mc, vc

# ---- correr en la muestra: una foto por nivel (0 a 4) ----
archivos = sorted(glob.glob(os.path.join(CARPETA, "**", "*.png"), recursive=True))
print("Fotos encontradas:", len(archivos))
elegidas = []
for nivel in range(5):
    for f in archivos:
        if os.path.basename(f).startswith(f"nivel{nivel}"):
            elegidas.append(f); break
if len(elegidas) < 5:
    elegidas = archivos[:5]   # por si los nombres no empiezan con nivelN

fig, ax = plt.subplots(5, 3, figsize=(9, 15))
for i, f in enumerate(elegidas):
    img = leer(f)
    cuad, mc, vc = preparar(img)
    print(os.path.basename(f), "| tamano original:", img.shape[1], "x", img.shape[0],
          "| retina valida:", int(mc.sum()), "px de", mc.size, f"({mc.sum()/mc.size:.0%})")
    for j, (im, t) in enumerate([(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original"),
                                 (cv2.cvtColor(cuad, cv2.COLOR_BGR2RGB), "Retina 512x512"),
                                 (vc, "Verde con CLAHE")]):
        ax[i, j].imshow(im, cmap="gray" if j == 2 else None)
        ax[i, j].set_title(t, fontsize=9); ax[i, j].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(RAIZ, "figuras", "fig_fase3_paso1_recorte.png"), dpi=300)
print("Figura guardada en figuras\\fig_fase3_paso1_recorte.png")
