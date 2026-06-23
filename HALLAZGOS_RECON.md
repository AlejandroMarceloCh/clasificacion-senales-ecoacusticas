# Recon multi-agente — CS3061 Proyecto 2 (Eco-Acústico)

Síntesis de 10 análisis paralelos sobre el dataset real (train 1906×68, test 477×68).
Todos los números fueron medidos por agentes corriendo código, no estimados.

## 0. El insight que define el proyecto
**La tarea es genuinamente DIFÍCIL.** Baseline F1-macro ≈ 0.41 (LDA/GradientBoosting),
techo realista ≈ 0.45–0.55 incluso con XGBoost. **NO esperar 0.85.**
- Separación binaria ranas vs aves: alta (F1 ≈ 0.79).
- Pero multiclase es difícil: las ranas se confunden con aves (Lepto→Pheuc 50.7% error, Fisher ratio 0.225 = casi indistinguibles).
- **Consecuencia:** la nota NO se juega en "lograr accuracy alto", se juega en el **análisis crítico** — que es justo lo que premia la rúbrica. Reportar F1-macro honesto + matriz de confusión bien interpretada vale más que un número inflado.

## 1. Decisiones de datos (resueltas por evidencia)
| Decisión | Veredicto | Por qué (medido) |
|---|---|---|
| `is_tp` | **Usar todo (1906)**, no filtrar | Filtrar deja 252 obs (−87%), empeora desbalance, overfitting. is_tp aporta solo +0.6% F1 (redundante con mel). |
| `songtype_id` | **INCLUIR como feature** | Aporta **+9.4% F1** (de 0.41 a 0.51). Hallazgo no anticipado. songtype=4 solo tiene especies 17 y 23. |
| `recording_id` | **EXCLUIR siempre** | Leakage: 26.9% de los IDs de test están en train. |
| Escalado | **StandardScaler en pipeline sklearn** | Multicolinealidad severa (113 pares \|r\|>0.9, condition number 33,685). Sin escalar, PCA/DBSCAN se sesgan. Fit en train, aplicar en test (sin fuga). |
| Plantilla LaTeX | No hay oficial UTEC → **article 11pt estándar** | Confirmado por agente de formato. |

## 2. Estrategia por criterio de rúbrica
- **C1 (3pts):** diagrama de arquitectura CSV→inferencia + fundamentar X∈ℝ⁶⁴ + mencionar desbalance 2:1.
- **C2 (5pts):** PCA (7 comp = 91% var, <1ms) + t-SNE sobre PCA(20) (4.8s). **Reportar TIEMPOS en tabla** (omitirlos = cae a 2/5). UMAP opcional (+0.5pt). Scatter 2D coloreado por especie.
- **C3 (5pts):** GMM vs DBSCAN (2 paradigmas). Silhouette por k=2..8. **Justificar k por métricas, NO por "hay 5 clases"** (= cae a 2.5/5). Insight clave: clustering NO reconstruye taxonomía (ARI=0.024) — reportarlo como hallazgo, no como fallo.
- **C4 (5pts):** MLP (64→256→128→64→5, ReLU+BatchNorm+Dropout, CrossEntropy ponderado) vs XGBoost/GradientBoosting. **F1-macro como métrica primaria** (usar Accuracy = cae a 2/5). StratifiedKFold + class_weight='balanced' obligatorios. 2 matrices de confusión 5×5. Loss function formal escrita. Esperado: GB ~0.43, MLP ~0.38-0.42.
- **C5 (2pts):** trade-off latencia vs F1 (XGBoost ~25ms/0.45 vs MLP ~8ms/0.40) + política 3 zonas medida: Verde(P≥85%)≈11%, Amarillo≈77%, Rojo≈12%. Redacción en voz pasiva impersonal.
- **Bonus Streamlit (+2):** ~8h, app.py ~120 líneas, escenarios precargados con recording_ids reales del test + semáforo. Solo si base ≥18pts y queda tiempo. ROI dudoso.

## 3. Trampas de penalización (automáticas — NO discrecionales)
- 🔴 **Fuente <14 en figuras = −3** (la más grave; default matplotlib es 10pt). Fijar `plt.rcParams` global y verificar en el PDF.
- 🔴 C2 sin tiempos / C3 sin Silhouette / C4 con Accuracy → cada uno tumba la sección a ~40%.
- 🟠 Sin repo GitHub público −1 · Sin tabla coevaluación (§3.6) −1 · >10 páginas −1 · code listing >15 líneas −1 · errores LaTeX −0.5.

## 4. Pipeline técnico recomendado (orden)
1. Preprocesamiento: drop recording_id, encode songtype_id, StandardScaler (en Pipeline).
2. C2: PCA + t-SNE con tiempos.
3. C3: GMM + DBSCAN + Silhouette.
4. C4: MLP (PyTorch) vs XGBoost, StratifiedKFold, F1-macro, matrices.
5. C5: umbrales sobre softmax + trade-off.
6. (Bonus) Streamlit.
7. Informe LaTeX ≤10pág + repo público + coevaluación.
