import os, hashlib
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

RUTA = r"C:\Users\Jhonatan\Documents\IA ..... 1\PROYECTO INTEGRADOR\Dataset aptos"

# 1) Leo los tres CSV y les agrego su origen y la ruta relativa de la imagen (carpeta\carpeta\id.png)
grupos = [("train", "train_1.csv", "train_images"),
          ("val", "valid.csv", "val_images"),
          ("test", "test.csv", "test_images")]
tablas = []
for origen, csv, carpeta in grupos:
    t = pd.read_csv(os.path.join(RUTA, csv))
    t["origen"] = origen
    t["archivo"] = t["id_code"].apply(lambda i: os.path.join(carpeta, carpeta, i + ".png"))
    tablas.append(t)
df = pd.concat(tablas, ignore_index=True)
print("Total de filas:", len(df))

# 2) Etiqueta binaria: 0 = sin retinopatía, 1 = con retinopatía (niveles 1 a 4)
df["y"] = (df["diagnosis"] > 0).astype(int)

# 3) Controles de integridad
print("id_code repetidos entre los 3 grupos:", df["id_code"].duplicated().sum())
existe = df["archivo"].apply(lambda a: os.path.isfile(os.path.join(RUTA, a)))
print("Imágenes que no existen en disco:", (~existe).sum())

# 4) Imágenes dañadas, tamaños y hash MD5 (para detectar imágenes idénticas con otro nombre)
corruptas, tamanos, hashes = [], [], []
for k, a in enumerate(df["archivo"]):
    ruta = os.path.join(RUTA, a)
    try:
        with Image.open(ruta) as im:
            im.verify()
        with Image.open(ruta) as im:
            tamanos.append(im.size)
        with open(ruta, "rb") as f:
            hashes.append(hashlib.md5(f.read()).hexdigest())
    except Exception:
        corruptas.append(a); hashes.append(None)
    if (k + 1) % 500 == 0:
        print("  revisadas", k + 1, "de", len(df))
df["md5"] = hashes
print("Imágenes corruptas:", len(corruptas))
print("Imágenes con contenido idéntico a otra (MD5):", df["md5"].duplicated().sum())
print("Tamaños distintos:", len(set(tamanos)), "| ejemplos:", sorted(set(tamanos))[:5])

# 5) Distribución de clases: total y por grupo de origen
orig = df["diagnosis"].value_counts().sort_index()
binar = df["y"].value_counts().sort_index()
print("\nDistribución original:\n", pd.DataFrame({"n": orig, "%": (orig / len(df) * 100).round(2)}))
print("\nDistribución binaria:\n", pd.DataFrame({"n": binar, "%": (binar / len(df) * 100).round(2)}))
print("\n% con RD por grupo de origen:\n", (df.groupby("origen")["y"].mean() * 100).round(2))

# 6) Figura a 300 DPI para el informe
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(orig.index.astype(str), orig.values, color="#4C72B0")
ax[0].set_title("Distribución original (niveles 0 a 4)")
ax[0].set_xlabel("Nivel de retinopatía"); ax[0].set_ylabel("Número de imágenes")
ax[1].bar(["Sin RD (0)", "Con RD (1)"], binar.values, color=["#55A868", "#C44E52"])
ax[1].set_title("Distribución binaria"); ax[1].set_ylabel("Número de imágenes")
plt.tight_layout()
plt.savefig("figuras/fig_fase1_distribucion_clases.png", dpi=300)

# 7) Guardo la tabla de etiquetas (sin rutas de tu PC) para las siguientes fases
df[["id_code", "diagnosis", "y", "origen", "archivo", "md5"]].to_csv("datos/etiquetas_binarias.csv", index=False)
print("\nGuardado: datos/etiquetas_binarias.csv y figuras/fig_fase1_distribucion_clases.png")
