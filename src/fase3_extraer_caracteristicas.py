# src\fase3_extraer_caracteristicas.py
# FASE 3 - PASO 6: calcula las 9 caracteristicas de las 3,530 imagenes y guarda datos\caracteristicas.csv
import os, sys, time, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retina_utils import (leer, preparar, disco_optico, vasos, lesiones_rojas,
                          exudados, densidad_vasos, textura_glcm)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA = r"C:\Users\Jhonatan\Documents\IA ..... 1\PROYECTO INTEGRADOR\Dataset aptos"   # carpeta de las imagenes
CSV_ENTRADA = os.path.join(RAIZ, "datos", "etiquetas_binarias_limpio.csv")
CSV_SALIDA = os.path.join(RAIZ, "datos", "caracteristicas.csv")
CSV_ERRORES = os.path.join(RAIZ, "resultados", "fotos_con_error.csv")

def extraer(fila):
    # fila = (id_code, ruta_relativa). Devuelve un dict con las 9 caracteristicas (o el error)
    warnings.filterwarnings("ignore")
    id_code, rel = fila
    try:
        ruta = os.path.join(RUTA, *rel.replace("\\", "/").split("/"))
        img = leer(ruta)
        if img is None:
            raise ValueError("no se pudo leer la imagen")
        cuad, mc, vc = preparar(img)                       # Paso 1
        d, _ = disco_optico(cuad, mc)                      # Paso 3a
        vas = vasos(vc, mc)                                # Paso 3b
        rojas, n_r = lesiones_rojas(vc, mc, d, vas)        # Paso 2
        exu, n_e = exudados(vc, mc, d, vas)                # Paso 3c
        _, dens = densidad_vasos(vc, mc, d)                # Paso 4
        tex = textura_glcm(vc, mc, d)                      # Paso 5
        valida = int(((mc > 0) & (d == 0)).sum())          # retina sin disco
        r = {"id_code": id_code,
             "area_rojas": rojas.sum() / valida * 100, "n_rojas": n_r,
             "area_exudados": exu.sum() / valida * 100, "n_exudados": n_e,
             "densidad_vasos": dens}
        r.update(tex)
        return r
    except Exception as e:
        return {"id_code": id_code, "error": str(e)}

if __name__ == "__main__":
    df = pd.read_csv(CSV_ENTRADA)
    tareas = list(zip(df["id_code"], df["archivo"]))
    nucleos = max(1, (os.cpu_count() or 2) - 1)
    print(f"Imagenes: {len(tareas)} | nucleos usados: {nucleos}")
    t0 = time.time(); salida = []
    with ProcessPoolExecutor(max_workers=nucleos) as ex:
        for i, r in enumerate(ex.map(extraer, tareas, chunksize=8), 1):
            salida.append(r)
            if i % 100 == 0 or i == len(tareas):
                print(f"  {i}/{len(tareas)} | {time.time()-t0:.0f} s")
    res = pd.DataFrame(salida)
    if "error" in res.columns:
        errores = res[res["error"].notna()][["id_code", "error"]]
        res = res[res["error"].isna()].drop(columns="error")
    else:
        errores = pd.DataFrame(columns=["id_code", "error"])
    errores.to_csv(CSV_ERRORES, index=False)
    final = df[["id_code", "diagnosis", "y", "origen"]].merge(res, on="id_code", how="inner")
    final.to_csv(CSV_SALIDA, index=False)
    print(f"\nGuardado: {CSV_SALIDA} | filas: {len(final)} | con error: {len(errores)}")
    print(final.drop(columns=["id_code"]).describe().round(3).T.to_string())
