"""Reduccion de dimensionalidad (C2): PCA vs t-SNE (+UMAP si esta)."""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from style import savefig, save_facts
from data import get_data, MEL_COLS, TIPO
from s1_eda import COLORS


def run():
    X_train, X_test, y_train, y_test, scaler = get_data()
    X = X_train[MEL_COLS].to_numpy()
    tabla = []

    # PCA
    t0 = time.perf_counter()
    pca_full = PCA().fit(X)
    t_pca = time.perf_counter() - t0
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n90 = int(np.argmax(cumvar >= 0.90) + 1)
    n95 = int(np.argmax(cumvar >= 0.95) + 1)
    Xp2 = PCA(n_components=2).fit_transform(X)
    tabla.append(("PCA(2)", round(t_pca, 4), round(float(cumvar[1])*100, 1)))
    tabla.append((f"PCA({n90})", round(t_pca, 4), 90.0))

    # scree + varianza acumulada
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.bar(range(1, 21), pca_full.explained_variance_ratio_[:20]*100, color="#2a9d8f")
    a1.set_xlabel("Componente principal"); a1.set_ylabel("Varianza explicada (%)")
    a1.set_title("Scree plot (primeras 20 PC)")
    a2.plot(range(1, len(cumvar)+1), cumvar*100, lw=2, color="#264653")
    a2.axhline(90, ls="--", color="#e76f51"); a2.axvline(n90, ls="--", color="#e76f51")
    a2.set_xlabel("N.º de componentes"); a2.set_ylabel("Varianza acumulada (%)")
    a2.set_title(f"{n90} PC -> 90% | {n95} PC -> 95%")
    savefig(fig, "pca_varianza.png")

    # t-SNE sobre PCA(20) para eficiencia
    Xp20 = PCA(n_components=20).fit_transform(X)
    t0 = time.perf_counter()
    Xt = TSNE(n_components=2, init="pca", perplexity=30, random_state=42).fit_transform(Xp20)
    t_tsne = time.perf_counter() - t0
    tabla.append(("t-SNE(2)", round(t_tsne, 3), None))

    # UMAP opcional
    Xu = None; t_umap = None
    try:
        from umap import UMAP
        t0 = time.perf_counter()
        Xu = UMAP(n_components=2, n_neighbors=15, random_state=42).fit_transform(Xp20)
        t_umap = round(time.perf_counter() - t0, 3)
        tabla.append(("UMAP(2)", t_umap, None))
    except Exception:
        pass

    # scatter comparativo
    proys = [("PCA", Xp2), ("t-SNE", Xt)] + ([("UMAP", Xu)] if Xu is not None else [])
    fig, axs = plt.subplots(1, len(proys), figsize=(6*len(proys), 5.5))
    if len(proys) == 1:
        axs = [axs]
    for ax, (nom, P) in zip(axs, proys):
        for k in [10, 12, 17, 18, 23]:
            m = y_train == k
            ax.scatter(P[m, 0], P[m, 1], s=12, alpha=0.6, color=COLORS[k], label=f"{k} ({TIPO[k]})")
        ax.set_title(f"Proyeccion {nom}"); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    axs[-1].legend(markerscale=1.5, loc="best")
    savefig(fig, "proyecciones_2d.png")

    facts = save_facts({
        "pca_n90": n90, "pca_n95": n95,
        "pca_var_pc1": round(float(pca_full.explained_variance_ratio_[0])*100, 1),
        "tiempos_reduccion": {"PCA_s": round(t_pca, 4), "tSNE_s": round(t_tsne, 3),
                              "UMAP_s": t_umap},
        "umap_disponible": Xu is not None,
        "tabla_reduccion": [[a, b, c] for a, b, c in tabla],
    })
    assert cumvar[n90-1] >= 0.90
    assert any(t[0].startswith("t-SNE") for t in tabla), "falta t-SNE con tiempo"
    print("PCA 90% var ->", n90, "comp | t-SNE", round(t_tsne, 2), "s | UMAP:", Xu is not None)
    print("OK")


if __name__ == "__main__":
    run()
