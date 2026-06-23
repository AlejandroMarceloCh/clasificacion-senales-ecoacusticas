# PLAN.md — CS3061 Proyecto 2: Clasificación Eco-Acústica

**Objetivo global:** pipeline ML completo que clasifica 5 especies (2 ranas + 3 aves)
a partir de 64 features MFCC, documentado en informe LaTeX (≤10 pág) + repo GitHub.
Rúbrica: 20 pts base (+2 bonus Streamlit).

**Entregables:** informe `.tex` · repo GitHub público · (opcional) app Streamlit.

**Regla de oro de cada sprint:** no se cierra hasta que su **test** pase.
Para sprints de ML el "test" es un `assert` sobre métricas/shapes + verificación de
que toda figura tenga **fuente ≥14** (penalización −3 si no).

---

## Oráculo (números medidos en la recon — sirven de referencia/test)
| Métrica | Valor esperado |
|---|---|
| Train / Test | 1906×68 / 477×68 |
| Baseline F1-macro (solo mel) | ≈ 0.41 |
| F1-macro con songtype_id | ≈ 0.51 |
| PCA para 90% varianza | 7 componentes |
| Overlap recording_id train↔test | 26.9% (→ excluir) |
| Techo realista F1-macro | 0.45–0.55 (NO 0.85) |

---

## Sprint 0 — Setup + Preprocesamiento
**Goal:** proyecto ordenado, datos cargados, pipeline de features sin leakage.
- Mover proyecto a `~/Desktop/PROYECTOS_2026/`, crear `venv/`, `git init` + repo GitHub.
- Cargar train/test, separar X (64 mel + songtype_id) / y (species_id).
- Pipeline sklearn: drop `recording_id`, one-hot `songtype_id`, `StandardScaler` (fit solo en train).

**Test:**
```python
assert X_train.shape[0] == 1906 and X_test.shape[0] == 477
assert 'recording_id' not in X_train.columns          # excluido (leakage)
assert abs(X_train_scaled.mean()) < 0.01              # scaler aplicado
# scaler NO fue fiteado con test:
assert scaler.n_samples_seen_ == 1906
print("✅ Sprint 0 OK")
```

---

## Sprint 1 — EDA + Diagrama de arquitectura (C1, 3 pts)
**Goal:** figuras EDA legibles + el diagrama de arquitectura entregable.
- Distribución de clases (barras, desbalance 2:1), tabla de stats de las mel.
- **Diagrama de arquitectura** del pipeline (CSV→preproc→reducción→clustering→clasif→umbrales→inferencia).
  Herramienta: **Excalidraw** (export PNG) o **TikZ** si se quiere nativo en LaTeX. Decidir y producirlo, no dejarlo "para después".
- **[Hueco #6]** el diagrama es entregable con peso de C1 → debe existir como archivo, no como idea.

**Test:**
```python
import matplotlib
for fig in figuras_generadas:
    for ax in fig.axes:
        assert ax.xaxis.label.get_size() >= 14         # fuente ≥14 (anti -3pts)
assert os.path.exists('figs/balance_clases.png')
assert os.path.exists('figs/arquitectura_pipeline.png')   # [#6] diagrama producido
print("✅ Sprint 1 OK")
```

---

## Sprint 2 — Reducción de dimensionalidad (C2, 5 pts)
**Goal:** PCA vs t-SNE (UMAP opcional) con **tiempos** y varianza.
- PCA: scree + varianza acumulada + scatter 2D coloreado por especie.
- t-SNE sobre PCA(20). Intentar instalar UMAP; si falla, documentarlo.
- Tabla: Método | Tiempo(s) | Varianza% | Interpretabilidad.

**Test:**
```python
assert cumvar[7] >= 0.90                                # 7 comp ≥90% var
assert 'tiempo_s' in tabla_comparativa.columns          # TIEMPOS reportados (clave C2)
assert scatter_pca.exists() and scatter_tsne.exists()
print("✅ Sprint 2 OK")
```

---

## Sprint 3 — Clustering (C3, 5 pts)
**Goal:** 2 paradigmas (GMM vs DBSCAN) + Silhouette, k justificado por métrica.
- Silhouette para k=2..8 (KMeans/GMM), k-distance plot para eps de DBSCAN.
- Reportar ARI vs especies (≈0.02) como hallazgo: clustering ≠ taxonomía.

**Test:**
```python
assert len(silhouette_por_k) >= 6                       # barrido k=2..8
assert {'GMM','DBSCAN'} <= set(metodos_usados)          # 2 paradigmas
assert k_elegido_justificado_por == 'silhouette'        # NO "porque hay 5 clases"
print("✅ Sprint 3 OK")
```

---

## Sprint 4 — MLP vs Ensamble (C4, 5 pts)
**Goal:** dos modelos comparados con F1-macro + matrices de confusión, con todo lo formal que pide la rúbrica.
- MLP en **PyTorch** (64→256→128→64→5, ReLU+BatchNorm+Dropout, CrossEntropy ponderado).
- XGBoost/GradientBoosting con `class_weight`/`scale_pos_weight`.
- StratifiedKFold. 2 matrices de confusión 5×5.
- **[Hueco #1] Loss function escrita en fórmula** (Cross-Entropy ponderada, LaTeX) — no solo nombrarla.
- **[Hueco #2] Tabla de topología** de la red (capa, neuronas, activación, dropout) como artefacto.
- **[Hueco #3] Experimento de POSICIÓN de Dropout/BatchNorm:** entrenar ≥3 configs (BN→Dropout, Dropout→BN, sin reg) y graficar sus curvas Loss/época para comparar estabilidad.
- **[Hueco #4] Evaluar en el `test.csv` oficial (477 con labels)**, no solo CV — reportar F1-macro de test.

**Test:**
```python
assert metrica_primaria == 'f1_macro'                   # NO accuracy (clave C4)
assert 0.35 <= f1_macro_mlp <= 0.60                     # rango realista
assert conf_matrix_mlp.shape == (5,5) and conf_matrix_xgb.shape == (5,5)
assert usa_stratified_kfold and usa_class_weight
assert 'loss_formula_tex' in facts                      # [#1] fórmula escrita
assert os.path.exists('figs/topologia_tabla.tex') or 'topologia' in facts   # [#2]
assert len(curvas_dropout_bn_configs) >= 3              # [#3] experimento posición
assert 'f1_macro_test_oficial' in facts                 # [#4] evaluado en test.csv
print("✅ Sprint 4 OK")
```

---

## Sprint 5 — Umbrales + Trade-offs (C5, 2 pts)
**Goal:** política de 3 zonas sobre softmax + trade-off latencia/F1.
- Zonas: Verde P≥85% (auto), Amarillo 40–85% (auditoría), Rojo <40% (descarte).
- Medir % de registros por zona + tabla latencia(ms) vs F1 (MLP vs XGBoost).

**Test:**
```python
zonas = clasificar_por_umbral(probas)
assert set(zonas.unique()) <= {'verde','amarillo','rojo'}
assert abs(sum(pct_por_zona.values()) - 1.0) < 0.01     # cobertura total
assert 'latencia_ms' in tabla_tradeoff.columns
print("✅ Sprint 5 OK")
```

---

## Sprint 6 — Streamlit (bonus +2) [OPCIONAL]
**Goal:** app con escenarios precargados (recording_ids reales) + semáforo de zonas.
**Decisión:** solo si la base ≥18 pts y quedan ≥8h. ROI dudoso.

**Test:**
```python
# app.py corre sin error, carga modelo, predice
assert modelo_y_scaler_serializados                     # joblib, incluye scaler
assert tiempo_inferencia_ms < 100
print("✅ Sprint 6 OK")
```

---

## Sprint 7 — Informe LaTeX + Entrega
**Goal:** informe ≤10 pág, repo público, tabla de coevaluación.
- 6 secciones (3.1–3.6), tono académico impersonal, code listings ≤15 líneas.
- Tabla §3.6 (integrantes + % + tareas, suma 100%).

**Test (checklist final):**
```python
assert num_paginas_pdf <= 10                            # −1 si excede
assert compila_sin_errores_latex
assert repo_github_es_publico                           # (requiere OK del usuario para push)
assert tabla_coevaluacion_completa and suma_pct == 100
assert todas_figuras_fuente >= 14                       # −3 si no
# [Hueco #5] toda figura/tabla numerada, con caption autónomo y CITADA en el texto:
assert todas_las_figuras_citadas_en_texto               # \ref{} para cada \label{}
print("✅ Sprint 7 OK — listo para entregar")
```

---

## Definición de "Hecho" (global)
- [ ] Los 7 sprints con su test en verde (Sprint 6 opcional).
- [ ] Informe ≤10 pág, todas las figuras fuente ≥14, sin errores LaTeX.
- [ ] Repo GitHub público + tabla de coevaluación.
- [ ] F1-macro reportado honesto (no inflado) con matrices de confusión interpretadas.

## Fuera de scope (a propósito)
- Deep learning audio crudo (el dataset ya viene en features MFCC).
- Buscar accuracy >0.85 (techo real ~0.55; el valor está en el análisis).
- Feature engineering sobre `recording_id` (leakage).
