# src\retina_utils.py
# Funciones compartidas de la Fase 3 (se reutilizan en todos los pasos)
import cv2, numpy as np

LADO = 512

def leer(ruta):
    # cv2.imread falla con rutas con caracteres raros, por eso se lee asi
    return cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)

def preparar(img):
    # PASO 1: devuelve (retina 512x512 a color, mascara de retina, canal verde con CLAHE)
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    m = (g > 10).astype(np.uint8)                       # retina = pixel > 10
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])         # region conectada mas grande
    m = (lab == k).astype(np.uint8)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m); cv2.drawContours(m, cnt, -1, 1, -1)   # rellena huecos
    x, y, w, h = cv2.boundingRect(m)
    img, m = img[y:y+h, x:x+w], m[y:y+h, x:x+w]
    L = max(w, h); top, left = (L-h)//2, (L-w)//2
    cuad = np.zeros((L, L, 3), np.uint8); mc = np.zeros((L, L), np.uint8)
    cuad[top:top+h, left:left+w] = img; mc[top:top+h, left:left+w] = m
    cuad = cv2.resize(cuad, (LADO, LADO), interpolation=cv2.INTER_AREA)
    mc = cv2.resize(mc, (LADO, LADO), interpolation=cv2.INTER_NEAREST)
    mc = cv2.erode(mc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cuad = cuad * mc[:, :, None]
    vc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cuad[:, :, 1]) * mc
    return cuad, mc, vc

def lesiones_rojas(vc, mc, d=None, vas=None, T=20, K=15, amin=6, amax=600, ecc=0.95, sol=0.6):
    # PASO 2: manchas oscuras y compactas (microaneurismas / hemorragias)
    from skimage.measure import label, regionprops
    vs = cv2.GaussianBlur(vc, (5, 5), 1.5)                         # suaviza ruido
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K))
    bh = cv2.morphologyEx(vs, cv2.MORPH_BLACKHAT, se).astype(np.int16)  # cuanto mas oscuro que su entorno
    borde = cv2.erode(mc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    m = ((bh > T) & (borde > 0)).astype(np.uint8)
    if d is not None: m[d > 0] = 0                                # excluye disco optico
    if vas is not None: m[vas > 0] = 0                            # excluye vasos
    lab = label(m); out = np.zeros_like(m); n = 0
    for r in regionprops(lab):                                    # filtro por forma
        if amin <= r.area <= amax and r.eccentricity < ecc and r.solidity > sol:
            out[lab == r.label] = 1; n += 1                       # compacta = lesion; alargada = vaso
    return out, n

def disco_optico(cuad, mc, R=40):
    # PASO 3a: disco optico = region brillante, redonda y compacta (canal rojo suavizado)
    from skimage.measure import label, regionprops
    r = cv2.GaussianBlur(cuad[:, :, 2], (0, 0), 4).astype(np.float32); r[mc == 0] = 0
    vals = r[mc > 0]; centro = None
    for p in (99.5, 99, 98, 97, 96, 95, 93, 90, 85):               # baja el umbral hasta hallar una region redonda
        b = ((r >= np.percentile(vals, p)) & (mc > 0)).astype(np.uint8)
        cand = [(q.area, q.centroid) for q in regionprops(label(b))
                if 1200 <= q.area <= 12000 and q.solidity > 0.9 and q.eccentricity < 0.75]
        if cand:
            _, (cy, cx) = max(cand); centro = (int(cx), int(cy)); break
    if centro is None:                                             # plan B: el punto mas brillante
        rr = cv2.GaussianBlur(cuad[:, :, 2], (0, 0), 12).astype(np.float32); rr[mc == 0] = 0
        y, x = np.unravel_index(np.argmax(rr), rr.shape); centro = (int(x), int(y))
    d = np.zeros_like(mc); cv2.circle(d, centro, R, 1, -1)         # circulo de exclusion de radio R
    return d, centro

def vasos(vc, mc, pct=90):
    # PASO 3b: mascara de vasos con el filtro de Frangi (detecta estructuras largas y oscuras)
    from skimage.filters import frangi
    from skimage.measure import label, regionprops
    f = frangi(vc.astype(np.float32) / 255., sigmas=(1.5, 2, 3), black_ridges=True)
    f[mc == 0] = 0
    v = (f > np.percentile(f[mc > 0], pct)).astype(np.uint8)
    lab = label(v); keep = np.zeros_like(v)
    for r in regionprops(lab):
        if r.area >= 120: keep[lab == r.label] = 1                 # vasos = trazos largos conectados
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

def exudados(vc, mc, d, vas, T=70, K=31, amin=8):
    # PASO 3c: manchas brillantes (top-hat blanco) fuera del disco y de los vasos
    from skimage.measure import label, regionprops
    vs = cv2.GaussianBlur(vc, (5, 5), 1.0)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K))
    th = cv2.morphologyEx(vs, cv2.MORPH_TOPHAT, se).astype(np.int16)   # cuanto mas brillante que su entorno
    borde = cv2.erode(mc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    m = ((th > T) & (borde > 0) & (d == 0) & (vas == 0)).astype(np.uint8)
    lab = label(m); out = np.zeros_like(m); n = 0
    for r in regionprops(lab):
        if r.area >= amin:
            out[lab == r.label] = 1; n += 1
    return out, n

def densidad_vasos(vc, mc, d, frac=0.25):
    # PASO 4: largo total de vasos (esqueleto de 1 px) / retina valida * 100
    from skimage.filters import frangi
    from skimage.morphology import skeletonize, remove_small_objects
    f = frangi(vc.astype(np.float32) / 255., sigmas=(1.5, 2, 3), black_ridges=True)
    f[mc == 0] = 0
    valida = (mc > 0) & (d == 0)                              # retina sin disco optico
    umbral = frac * np.percentile(f[mc > 0], 99)              # umbral relativo a cada foto (corrige el contraste)
    b = remove_small_objects((f > umbral) & valida, 60)       # quita trazos pequenos (ruido)
    esq = skeletonize(b)                                      # vasos como lineas de 1 pixel
    return esq, esq.sum() / valida.sum() * 100

def textura_glcm(vc, mc, d, niveles=32):
    # PASO 5: textura GLCM (contraste, homogeneidad, energia, correlacion) solo dentro de la retina
    from skimage.feature import graycomatrix, graycoprops
    valida = (mc > 0) & (d == 0)                                   # retina sin disco optico
    q = (vc.astype(np.float32) / 256 * niveles).astype(np.uint8) + 1   # brillo 1..niveles
    q[~valida] = 0                                                 # fuera de la retina = 0 (se descarta)
    g = graycomatrix(q, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                     levels=niveles + 1, symmetric=True, normed=False)
    g = g[1:, 1:, :, :].astype(np.float64)                         # quita el nivel 0 (fondo)
    g = g / g.sum(axis=(0, 1), keepdims=True)                      # normaliza: probabilidades
    return {"glcm_contraste": float(graycoprops(g, "contrast").mean()),
            "glcm_homogeneidad": float(graycoprops(g, "homogeneity").mean()),
            "glcm_energia": float(graycoprops(g, "energy").mean()),
            "glcm_correlacion": float(graycoprops(g, "correlation").mean())}
