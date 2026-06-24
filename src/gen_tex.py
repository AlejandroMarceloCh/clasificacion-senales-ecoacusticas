import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REPORT = BASE / "report"
facts = json.loads((REPORT / "facts.json").read_text())

def num(x, dec=3):
    return f"{x:.{dec}f}" if isinstance(x, float) else str(x)

M = {
    "nTrain": facts["n_train"], "nTest": facts["n_test"], "nFeat": facts["n_features"],
    "ratioDesb": num(facts["ratio_desbalance"], 1), "paresCorr": facts["pares_corr_altos"],
    "pcaNotenta": facts["pca_n90"], "pcaNcincoa": facts["pca_n95"],
    "pcaVarPC": num(facts["pca_var_pc1"], 1),
    "tPca": num(facts["tiempos_reduccion"]["PCA_s"], 4),
    "tTsne": num(facts["tiempos_reduccion"]["tSNE_s"], 2),
    "tUmap": num(facts["tiempos_reduccion"]["UMAP_s"], 2),
    "ari": num(facts["ari_gmm5_vs_especies"], 3),
    "fMlpCV": num(facts["f1_macro_mlp_cv_mean"], 3), "fMlpTest": num(facts["f1_macro_mlp_test"], 3),
    "fXgbCV": num(facts["f1_macro_xgb_cv_mean"], 3), "fXgbTest": num(facts["f1_macro_xgb_test"], 3),
    "latMlp": num(facts["latencia_ms_mlp"], 4), "latXgb": num(facts["latencia_ms_xgb"], 4),
    "zVerde": num(facts["zonas_pct"]["verde"] * 100, 1),
    "zAmar": num(facts["zonas_pct"]["amarillo"] * 100, 1),
    "zRojo": num(facts["zonas_pct"]["rojo"] * 100, 1),
    "accVerde": num(facts["acc_zona_verde"] * 100, 1),
    "fSinFiltro": num(facts["f1_sin_filtro"], 3), "fFiltRojo": num(facts["f1_filtrando_rojo"], 3),
    "deltaMLP": num(facts["f1_macro_mlp_test"] - facts["f1_macro_xgb_test"], 3),
    "factorLat": int(round(facts["latencia_ms_xgb"] / facts["latencia_ms_mlp"])),
    "lossFormula": facts["loss_formula_tex"],
}
datos = "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in M.items()) + "\n"
(REPORT / "datos.tex").write_text(datos)

def tabla(colspec, header, filas, fname):
    body = "\n".join(filas)
    tex = (rf"\begin{{tabular}}{{{colspec}}}" "\n\\toprule\n"
           f"{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}")
    (REPORT / fname).write_text(tex)

filas = []
for met, t, var in facts["tabla_reduccion"]:
    v = f"{var:.1f}\\%" if var is not None else "---"
    estr = {"PCA(2)": "Global", "PCA(7)": "Global", "t-SNE(2)": "Local",
            "UMAP(2)": "Local + global"}.get(met, "")
    filas.append(f"{met} & {t} & {v} & {estr} \\\\")
tabla("lccc", "Método & Tiempo (s) & Varianza retenida & Estructura preservada",
      filas, "tab_reduccion.tex")

filas = []
for eps, nc, ruido, sil in facts["dbscan_tabla"]:
    s = f"${sil}$" if sil is not None else "---"
    filas.append(f"{eps} & {nc} & {ruido*100:.1f}\\% & {s} \\\\")
tabla("cccc", r"$\varepsilon$ & Clusters & \% ruido & Silhouette", filas, "tab_dbscan.tex")

filas = [
    rf"MLP (PyTorch)      & $\fMlpCV$ & $\mathbf{{\fMlpTest}}$ & $\latMlp$ \\",
    rf"XGBoost (ensamble) & $\fXgbCV$ & $\fXgbTest$ & $\latXgb$ \\",
]
tabla("lccc", "Modelo & F1-macro (CV) & F1-macro (test oficial) & Latencia (ms/muestra)",
      filas, "tab_modelos.tex")

print("Generado: datos.tex (%d macros), tab_reduccion.tex, tab_dbscan.tex, tab_modelos.tex" % len(M))
print("F1 MLP test (dinamico):", M["fMlpTest"])
