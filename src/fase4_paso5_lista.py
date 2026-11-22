# src\fase4_paso5_lista.py
# FASE 4 - PASO 5: deja registrada la lista final de caracteristicas (decision: se mantienen las 9, sin seleccion con y)
# No abre train.csv ni test.csv: solo reune lo ya calculado en los Pasos 2 y 3 (que usaron solo entrenamiento).
import os, pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
auc = pd.read_csv(os.path.join(RAIZ, "resultados", "auc_univariado_train.csv"))[["caracteristica", "AUC"]]
cam = pd.read_csv(os.path.join(RAIZ, "resultados", "auc_control_camara_train.csv"))[["caracteristica", "AUC_subconjunto", "pasa_control"]]

info = pd.DataFrame([
    ("area_rojas", "Lesiones rojas", "% de la retina cubierto por microaneurismas y hemorragias", "Se mantiene"),
    ("n_rojas", "Lesiones rojas", "Cantidad de manchas rojas", "Se mantiene (muy correlacionada con area_rojas)"),
    ("area_exudados", "Exudados", "% de la retina cubierto por exudados", "Se mantiene; sale al reves de lo esperado: se declara como limitacion"),
    ("n_exudados", "Exudados", "Cantidad de manchas de exudado", "Se mantiene; sale al reves de lo esperado: se declara como limitacion"),
    ("densidad_vasos", "Vasos", "Largo de vasos (esqueleto) / retina valida x 100", "Se mantiene (puede reflejar nitidez de la foto)"),
    ("glcm_contraste", "Textura GLCM", "Diferencia de brillo entre pixeles vecinos", "Se mantiene (correlacionada con homogeneidad y energia)"),
    ("glcm_homogeneidad", "Textura GLCM", "Parecido de brillo entre pixeles vecinos", "Se mantiene (correlacionada con contraste y energia)"),
    ("glcm_energia", "Textura GLCM", "Uniformidad de los pares de brillo (pocos patrones repetidos)", "Se mantiene (correlacionada con contraste)"),
    ("glcm_correlacion", "Textura GLCM", "Predictibilidad del brillo del vecino", "Se mantiene"),
], columns=["caracteristica", "familia", "que_mide", "decision"])

tab = info.merge(auc, on="caracteristica").merge(cam, on="caracteristica")
tab["nota_control_camara"] = tab["pasa_control"].map({True: "paso el control de camara", False: "no paso el control de camara (se declara como limitacion)"})
tab.to_csv(os.path.join(RAIZ, "resultados", "caracteristicas_finales.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(tab.round(3).to_string(index=False))
print("\nTotal de caracteristicas:", len(tab))
print("Guardado: resultados\\caracteristicas_finales.csv")
