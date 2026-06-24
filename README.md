# Clasificación de Señales Eco-Acústicas — CS3061 Proyecto 2 (Grupo 8)

Pipeline integral de Machine Learning que clasifica **5 especies** (2 anfibios + 3 aves)
a partir de 64 coeficientes MFCC (X ∈ ℝ⁶⁴). Curso CS3061 Machine Learning, UTEC.

## Resultados
| Modelo | F1-macro (CV) | F1-macro (test oficial) | Latencia |
|---|---|---|---|
| **MLP (PyTorch)** | 0.536 | **0.575** | 0.0009 ms/muestra |
| XGBoost | 0.489 | 0.536 | 0.0075 ms/muestra |

El MLP es el modelo de despliegue. Las especies cercanas se solapan acústicamente,
lo que limita el F1-macro alcanzable.

## Estructura
```
src/            data.py (preproc) · s1_eda · s2_reduccion · s3_clustering · s4_modelos · s5_umbrales
figs/           10 figuras (todas fuente ≥14)
report/         informe.tex + informe.pdf (4 págs) + facts.json + figs/
models/         mlp.pt · xgb_scaler.pkl
app.py          app Streamlit (semáforo de umbrales)
run_all.py      reproduce todo el pipeline
```

## Reproducir
```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 ./venv/bin/python run_all.py   # pipeline completo
cd report && tectonic informe.tex                                          # compilar informe
./venv/bin/streamlit run app.py                                            # app bonus
```
> Nota macOS: las env vars `KMP_DUPLICATE_LIB_OK`/`OMP_NUM_THREADS` evitan un segfault
> por la libomp duplicada entre PyTorch y XGBoost.

## Decisiones de ingeniería de datos
- **`recording_id` excluido**: 26.9% de IDs de test están en train (fuga de información).
- **`songtype_id` incluido** (one-hot): aporta +9.4% F1-macro.
- **`StandardScaler`** ajustado solo en train (sin contaminar el test).
- Métrica primaria **F1-macro** (no Accuracy) por el desbalance 2:1; `class_weight` balanceado.

## Pipeline por criterio de rúbrica
1. **C1** Diagrama de arquitectura + espacio vectorial ℝ⁶⁴.
2. **C2** PCA (7 comp = 90% var) vs t-SNE vs UMAP, con tiempos.
3. **C3** GMM vs DBSCAN + Silhouette; ARI=0.05 → el clustering no reconstruye la taxonomía.
4. **C4** MLP vs XGBoost, F1-macro, matrices de confusión, experimento de posición Dropout/BatchNorm.
5. **C5** Política de umbrales (Verde 24.5% / Amarillo 67.1% / Rojo 8.4%) + trade-off latencia/F1.
