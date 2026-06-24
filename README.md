# Clasificación de señales eco-acústicas (CS3061 - Proyecto 2)

Pipeline de Machine Learning para clasificar 5 especies (2 anfibios y 3 aves) a partir
de 64 coeficientes MFCC. Curso CS3061 Machine Learning, UTEC.

## Resultados
| Modelo | F1-macro (CV) | F1-macro (test) | Latencia |
|---|---|---|---|
| MLP (PyTorch) | 0.536 | 0.575 | 0.0009 ms/muestra |
| XGBoost | 0.489 | 0.536 | 0.0075 ms/muestra |

Usamos el MLP como modelo final. Hay especies que suenan muy parecido, así que el
F1-macro no llega muy alto.

## Estructura
```
src/        data.py, s1_eda, s2_reduccion, s3_clustering, s4_modelos, s5_umbrales
figs/       figuras del informe
report/     informe.tex, informe.pdf, facts.json
models/     mlp.pt, xgb_scaler.pkl
app.py      app de Streamlit
run_all.py  corre todo el pipeline
```

## Cómo correrlo
```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 ./venv/bin/python run_all.py
cd report && tectonic informe.tex
./venv/bin/streamlit run app.py
```
En Mac hay que poner las variables KMP_DUPLICATE_LIB_OK y OMP_NUM_THREADS para que no
se cuelgue por un choque entre las librerías de PyTorch y XGBoost.

## Notas sobre los datos
- Sacamos `recording_id` porque varios IDs del test también están en el train.
- Agregamos `songtype_id` como variable porque sube el F1.
- El `StandardScaler` se ajusta solo con el train.
- Usamos F1-macro y `class_weight` balanceado porque las clases están desbalanceadas.

## Qué hace cada etapa
1. Diagrama de la arquitectura y descripción de los datos de entrada.
2. Reducción de dimensionalidad con PCA, t-SNE y UMAP, con sus tiempos.
3. Clustering con GMM y DBSCAN usando Silhouette.
4. Modelos MLP y XGBoost comparados por F1-macro y matrices de confusión.
5. Umbrales de confianza (verde, amarillo y rojo) y comparación de latencia.
