# Comparación de regresión logística, KNN y árbol de decisión para clasificar retinopatía diabética en imágenes de fondo de ojo

Proyecto Integrador de **IS-484 Inteligencia Artificial I** (Ingeniería de Sistemas, UNSCH). Compara tres clasificadores clásicos (regresión logística, KNN y árbol de decisión) para separar fotos de fondo de ojo **sin retinopatía diabética** (nivel 0) de **con retinopatía diabética** (niveles 1 a 4), usando 9 características explicables que construí a partir de las imágenes. El PCA se evalúa como experimento complementario.

Autor: Jhonatan Conde Taquiri (GitHub: Taquiri05).

## Resultados principales
Conjunto de prueba: 706 fotos (347 con retinopatía, 359 sin), evaluado **una sola vez**.

| Modelo | Exactitud | Recall | F1 | AUC |
|---|---|---|---|---|
| Regresión logística (umbral 0.50) | 0.846 | 0.824 | 0.840 | 0.928 |
| Regresión logística (umbral 0.37) | 0.854 | 0.896 | 0.858 | 0.928 |
| KNN (K = 11) | 0.870 | 0.876 | 0.869 | 0.932 |
| Árbol de decisión | 0.863 | 0.885 | 0.864 | 0.910 |
| Línea base: solo el tamaño de la imagen | 0.897 | 0.945 | 0.900 | no aplica |

Intervalos de confianza al 95 % (bootstrap) en `resultados/fase6_tabla_comparativa.csv`.

Lectura de los resultados:
- Solo KNN frente a regresión logística muestra una diferencia apenas clara (F1 +0.029); KNN y árbol son equivalentes.
- **El tamaño de la imagen, sin mirar la retina, ya acierta más (0.897) que los modelos.** En APTOS 2019 cada cámara aporta casi una sola clase. En el subconjunto de una sola cámara (2588×1958, 103 fotos de prueba) la exactitud baja a 0.78 a 0.80.
- El nivel más leve (nivel 1) es el peor clasificado: recall de 0.52 a 0.73.
- El PCA no mejoró ningún modelo y no se usó en los modelos finales.

## Datos (las imágenes NO están en este repositorio)
Dataset de la competencia APTOS 2019 Blindness Detection (Kaggle). Sus reglas permiten usarlo para investigación académica y educación sin fines comerciales y prohíben duplicarlo, publicarlo o redistribuirlo. Por eso aquí solo hay código y tablas derivadas (`datos/` y `resultados/`). Para reproducir, descarga las imágenes de Kaggle y colócalas en una carpeta `Dataset aptos` junto a la carpeta del proyecto; revisa la ruta al inicio de cada script que lee imágenes.

De las 3,662 imágenes originales quedaron **3,530 únicas** tras eliminar duplicados (46 grupos cruzaban entrenamiento, validación y prueba).

## Fases y scripts
| Fase | Qué hace | Scripts (`src/`) | Salidas principales |
|---|---|---|---|
| 1 | Verifica el dataset y elimina duplicados | `fase1_verificar_dataset.py` | `datos/etiquetas_binarias_limpio.csv` |
| 2 | EDA y hallazgo del sesgo de cámara | `fase2_eda.py` | `datos/eda_imagenes.csv` |
| 3 | Extrae las 9 características | `fase3_extraer_caracteristicas.py` (`fase3_paso*.py` son pruebas de cada paso) | `datos/caracteristicas.csv` |
| 4 | División 80/20 sin fuga, control de cámara, escalado | `fase4_*.py`, `preprocesamiento.py` | `datos/train.csv`, `datos/test.csv` |
| 5 | Entrena los 3 modelos con validación cruzada | `fase5_*.py`, `validacion.py` | `resultados/configuraciones_elegidas.csv` |
| 6 | Evalúa en la prueba, controles de cámara | `fase6_*.py` | `resultados/fase6_*` |
| 7 | Importancia, análisis de errores | `fase7_*.py` | `resultados/fase7_*` |

Las 9 características: área y número de lesiones rojas, área y número de exudados, densidad de vasos y 4 texturas (contraste, homogeneidad, energía y correlación de GLCM).

## Notebook

El archivo `notebooks/Proyecto Integrador.ipynb` recorre las 17 etapas del proyecto (problema, datos, preparacion, modelos, evaluacion, interpretacion, limitaciones, etica y conclusiones). Usa el mismo codigo de `src/` y lee las tablas de `datos/` y `resultados/`. No vuelve a abrir el conjunto de prueba para decidir nada y no muestra fotos del dataset.

Para ejecutarlo: abrirlo desde la carpeta `notebooks/` con el entorno `.venv` activado y usar *Run All*.

## Cómo reproducir
Probado con Python 3.13.1 y scikit-learn 1.9.1.
1. Crear el entorno e instalar: `python -m venv .venv`, activarlo y `pip install -r requirements.txt`.
2. Ejecutar los scripts de `src/` en orden de fase, desde la carpeta raíz del proyecto (la fase 3 tarda unos 7 minutos con 15 núcleos).
3. `fase6_paso2_prueba.py` evalúa el conjunto de prueba **una sola vez** y deja un candado (`resultados/fase6_ejecutado.txt`); para repetirlo hay que borrar ese archivo a propósito.

## Metodología (resumen)
División estratificada 80/20 (`random_state=42`) antes de cualquier ajuste; logaritmo, escalado y PCA dentro de un `Pipeline` ajustado solo con entrenamiento; validación cruzada de 5 pliegues solo sobre entrenamiento, con el F1 medio como métrica de selección; prueba usada una sola vez.

## Alcance y limitaciones
Ejercicio académico de comparación de clasificadores: no es una herramienta de diagnóstico y no está validado clínicamente. Para interpretar los resultados:
- El dataset presenta sesgo de cámara (cada cámara aporta casi una sola clase), por eso se incluyen una línea base por tamaño de imagen y una evaluación en una sola cámara.
- El rendimiento es mejor en casos moderados y severos; el nivel 1 es el más difícil de clasificar.
- La textura se mide con características hechas a mano y puede depender de la nitidez de la foto.
- Se usó un solo dataset y una sola partición, sin validación externa.

