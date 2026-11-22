import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

RUTA = r"C:\Users\Jhonatan\Documents\IA ..... 1\PROYECTO INTEGRADOR\Dataset aptos"
ANCHO = 512   # todas las imágenes se reducen a este ancho para medir con la misma escala

def medir(archivo):
    # Leo la imagen (cv2.imdecode evita problemas con rutas con espacios o puntos)
    img = cv2.imdecode(np.fromfile(os.path.join(RUTA, archivo), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    p = cv2.resize(img, (ANCHO, int(round(h * ANCHO / w))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    return {"archivo": archivo, "ancho": w, "alto": h, "relacion": round(w / h, 3),
            "brillo": g.mean(), "rojo": p[:, :, 2].mean(), "verde": p[:, :, 1].mean(),
            "azul": p[:, :, 0].mean(), "borde_negro": (g < 15).mean(),
            "nitidez": cv2.Laplacian(g, cv2.CV_64F).var()}

if __name__ == "__main__":
    df = pd.read_csv("datos/etiquetas_binarias_limpio.csv")
    print("Midiendo", len(df), "imágenes...")
    filas = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as ex:
        for k, r in enumerate(ex.map(medir, df["archivo"], chunksize=20)):
            filas.append(r)
            if (k + 1) % 500 == 0:
                print("  medidas", k + 1)
    med = pd.DataFrame(filas)
    eda = df.merge(med, on="archivo")
    eda["tamano"] = eda["ancho"].astype(str) + "x" + eda["alto"].astype(str)
    eda.to_csv("datos/eda_imagenes.csv", index=False)

    # 1) Tamaño de imagen frente a la clase: ¿algún tamaño casi solo tiene una clase?
    t = eda.groupby("tamano")["y"].agg(n="size", pct_con_RD="mean").sort_values("n", ascending=False)
    t["pct_con_RD"] = (t["pct_con_RD"] * 100).round(1)
    print("\nTamaños más frecuentes y % con RD en cada uno:\n", t.head(12))
    t.to_csv("resultados/eda_tamano_vs_clase.csv")

    # 2) Medidas por clase (mediana)
    cols = ["brillo", "borde_negro", "nitidez", "relacion"]
    print("\nMediana por clase (0 = sin RD, 1 = con RD):\n", eda.groupby("y")[cols].median().round(3))

    # 3) Figura de cajas por clase (300 DPI)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, c, tt in zip(ax, ["brillo", "borde_negro", "nitidez"],
                        ["Brillo medio", "Fracción de borde negro", "Nitidez (var. Laplaciano)"]):
        a.boxplot([eda[eda.y == 0][c], eda[eda.y == 1][c]], labels=["Sin RD", "Con RD"], showfliers=False)
        a.set_title(tt)
    plt.tight_layout()
    plt.savefig("figuras/fig_fase2_medidas_por_clase.png", dpi=300)
    plt.close()

    # 4) Panel de ejemplos: 3 imágenes por nivel (filas = ejemplos, columnas = niveles 0 a 4)
    fig, ax = plt.subplots(3, 5, figsize=(15, 9.5))
    for n in range(5):
        ej = eda[eda.diagnosis == n].sample(3, random_state=42)
        for i, (_, f) in enumerate(ej.iterrows()):
            im = cv2.imdecode(np.fromfile(os.path.join(RUTA, f["archivo"]), np.uint8), cv2.IMREAD_COLOR)
            im = cv2.cvtColor(cv2.resize(im, (400, int(400 * im.shape[0] / im.shape[1]))), cv2.COLOR_BGR2RGB)
            ax[i, n].imshow(im); ax[i, n].axis("off")
            if i == 0:
                ax[i, n].set_title(f"Nivel {n}")
    plt.tight_layout()
    plt.savefig("figuras/fig_fase2_panel_niveles.png", dpi=300)
    print("\nGuardado: datos/eda_imagenes.csv, resultados/eda_tamano_vs_clase.csv y 2 figuras en figuras/")
