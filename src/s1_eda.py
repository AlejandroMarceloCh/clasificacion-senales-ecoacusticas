"""Sprint 1 — EDA + diagrama de arquitectura (C1)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from style import savefig, save_facts, FIGS
from data import load_raw, get_data, SPECIES, TIPO, MEL_COLS

COLORS = {10: "#2a9d8f", 12: "#264653", 17: "#e9c46a", 18: "#f4a261", 23: "#e76f51"}


def fig_balance(tr):
    vc = tr["species_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([f"{k}\n{TIPO[k]}" for k in vc.index], vc.values,
                  color=[COLORS[k] for k in vc.index])
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width()/2, v + 5, str(v), ha="center", fontsize=14)
    ax.set_ylabel("N.º de muestras")
    ax.set_xlabel("species_id")
    ax.set_title("Distribucion de clases (desbalance 2:1)")
    return savefig(fig, "balance_clases.png"), dict(vc)


def fig_corr(tr):
    corr = tr[MEL_COLS].corr().abs()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="magma", vmin=0, vmax=1)
    ax.set_title("Multicolinealidad |r| entre features MFCC")
    ax.set_xlabel("indice mel"); ax.set_ylabel("indice mel")
    cb = fig.colorbar(im, ax=ax); cb.ax.tick_params(labelsize=14)
    pares_altos = int((np.triu(corr.values, 1) > 0.9).sum())
    return savefig(fig, "correlacion_mel.png"), pares_altos


def fig_arquitectura():
    """Diagrama de arquitectura del pipeline (C1), generado con matplotlib (sin deps externas)."""
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")
    etapas = ["CSV\n(1906x68)", "Preproceso\nscaler+songtype", "Reduccion\nPCA/t-SNE",
              "Clustering\nGMM/DBSCAN", "Clasificacion\nMLP vs XGB", "Umbrales\n3 zonas", "Inferencia"]
    n = len(etapas); w = 1.0 / n
    for i, txt in enumerate(etapas):
        x = i * w
        ax.add_patch(plt.Rectangle((x + 0.008, 0.25), w - 0.016, 0.5,
                     facecolor="#264653" if i % 2 == 0 else "#2a9d8f", edgecolor="black"))
        ax.text(x + w/2, 0.5, txt, ha="center", va="center", color="white", fontsize=14)
        if i < n - 1:
            ax.annotate("", xy=(x + w + 0.002, 0.5), xytext=(x + w - 0.006, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Arquitectura del pipeline de clasificacion eco-acustica", fontsize=16)
    return savefig(fig, "arquitectura_pipeline.png")


if __name__ == "__main__":
    tr, te = load_raw()
    p_bal, balance = fig_balance(tr)
    p_corr, pares = fig_corr(tr)
    p_arq = fig_arquitectura()
    X_train, X_test, y_train, y_test, scaler = get_data()
    facts = save_facts({
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_features": int(X_train.shape[1]),
        "balance_train": {str(k): int(v) for k, v in balance.items()},
        "ratio_desbalance": round(max(balance.values()) / min(balance.values()), 2),
        "pares_corr_altos": pares,
        "especies": {str(k): v for k, v in SPECIES.items()},
    })
    # --- TEST Sprint 1 ---
    import os
    for f in ["balance_clases.png", "correlacion_mel.png", "arquitectura_pipeline.png"]:
        assert os.path.exists(FIGS / f), f"falta {f}"
    print("figuras:", os.listdir(FIGS))
    print("ratio desbalance:", facts["ratio_desbalance"], "| pares |r|>0.9:", pares)
    print("OK Sprint 1")
