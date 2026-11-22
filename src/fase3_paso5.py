# src\fase3_paso5.py
# FASE 3 - PASO 5: textura GLCM en la muestra (las 25 fotos)
import numpy as np, pandas as pd, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retina_utils import leer, preparar, disco_optico, textura_glcm

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
archivos = sorted(glob.glob(os.path.join(RAIZ, "muestra", "**", "*.png"), recursive=True))
print("Fotos encontradas:", len(archivos))

filas = []
for f in archivos:
    cuad, mc, vc = preparar(leer(f))
    d, _ = disco_optico(cuad, mc)
    r = textura_glcm(vc, mc, d)
    r["archivo"] = os.path.basename(f)
    r["nivel"] = int(os.path.basename(f)[5]) if os.path.basename(f).startswith("nivel") else -1
    filas.append(r)

df = pd.DataFrame(filas)
pd.set_option("display.width", 200)
print(df.round(4).to_string(index=False))
print("\nPromedio por nivel:")
print(df.groupby("nivel")[["glcm_contraste", "glcm_homogeneidad", "glcm_energia", "glcm_correlacion"]].mean().round(4))
df.to_csv(os.path.join(RAIZ, "resultados", "muestra_glcm.csv"), index=False)
print("\nGuardado: resultados\\muestra_glcm.csv")
