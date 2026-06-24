"""Clustering con KMeans, GMM y DBSCAN."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from style import savefig, save_facts
from data import get_data, MEL_COLS


def run():
    X_train, X_test, y_train, y_test, scaler = get_data()
    X = X_train[MEL_COLS].to_numpy()
    Xp = PCA(n_components=20, random_state=42).fit_transform(X)

    # Silhouette para cada k con KMeans y GMM
    ks = list(range(2, 9))
    sil_km, sil_gmm = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xp)
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              reg_covar=1e-4, random_state=42).fit(Xp)
        sil_km.append(silhouette_score(Xp, km.labels_))
        sil_gmm.append(silhouette_score(Xp, gmm.predict(Xp)))
    k_opt = ks[int(np.argmax(sil_gmm))]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, sil_km, "o-", lw=2, label="KMeans (centroide)", color="#2a9d8f")
    ax.plot(ks, sil_gmm, "s-", lw=2, label="GMM (probabilistico)", color="#e76f51")
    ax.axvline(k_opt, ls="--", color="gray")
    ax.set_xlabel("N.º de grupos (k)"); ax.set_ylabel("Coeficiente de Silhouette")
    ax.set_title("Seleccion de k por Silhouette"); ax.legend()
    savefig(fig, "silhouette_k.png")

    # DBSCAN para distintos eps
    dbscan_tab = []
    for eps in [2.5, 3.0, 3.5]:
        db = DBSCAN(eps=eps, min_samples=5).fit(Xp)
        lab = db.labels_
        n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
        ruido = float((lab == -1).mean())
        sil = silhouette_score(Xp, lab) if n_clusters > 1 else None
        dbscan_tab.append([eps, n_clusters, round(ruido, 3),
                           round(sil, 3) if sil is not None else None])

    # comparacion de los grupos con las especies reales (ARI)
    gmm5 = GaussianMixture(n_components=5, covariance_type="full",
                           reg_covar=1e-4, random_state=42).fit(Xp)
    ari = adjusted_rand_score(y_train, gmm5.predict(Xp))

    facts = save_facts({
        "silhouette_kmeans": [round(s, 3) for s in sil_km],
        "silhouette_gmm": [round(s, 3) for s in sil_gmm],
        "k_optimo_silhouette": int(k_opt),
        "dbscan_tabla": dbscan_tab,
        "ari_gmm5_vs_especies": round(float(ari), 3),
    })
    assert len(sil_gmm) >= 6, "barrido k incompleto"
    assert len(dbscan_tab) >= 2, "faltan eps de DBSCAN"
    print(f"k optimo (silhouette) = {k_opt} | ARI GMM(5) vs especies = {ari:.3f} (clustering != taxonomia)")
    print("DBSCAN eps/clusters/ruido/sil:", dbscan_tab)
    print("OK")


if __name__ == "__main__":
    run()
