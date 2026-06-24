"""Politica de umbrales y comparacion de costo y rendimiento."""
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
from sklearn.metrics import f1_score
from style import savefig, save_facts, BASE
from data import get_data
from s4_modelos import MLP, predict_mlp, y_to_idx, CLASES


def zona(p):
    if p >= 0.85:
        return "verde"
    if p >= 0.40:
        return "amarillo"
    return "rojo"


def run():
    X_train, X_test, y_train, y_test, scaler = get_data()
    Xte = X_test.to_numpy().astype(np.float32)
    yte_i = y_to_idx(y_test)
    proba = np.load(BASE / "models/proba_xgb_test.npy")
    pmax = proba.max(1)
    pred = proba.argmax(1)

    # politica de 3 zonas
    zonas = np.array([zona(p) for p in pmax])
    pct = {z: round(float((zonas == z).mean()), 3) for z in ["verde", "amarillo", "rojo"]}
    acc_verde = float((pred[zonas == "verde"] == yte_i[zonas == "verde"]).mean()) if (zonas=="verde").any() else 0.0
    # F1 al descartar zona roja (registros dudosos)
    keep = zonas != "rojo"
    f1_filtrado = f1_score(yte_i[keep], pred[keep], average="macro")
    f1_sin_filtro = f1_score(yte_i, pred, average="macro")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pmax, bins=30, color="#264653", alpha=0.85)
    ax.axvline(0.85, color="#2a9d8f", lw=2.5, ls="--", label="0.85 (verde)")
    ax.axvline(0.40, color="#e76f51", lw=2.5, ls="--", label="0.40 (rojo)")
    ax.set_xlabel("Probabilidad maxima (softmax)"); ax.set_ylabel("N.º de registros")
    ax.set_title("Distribucion de confianza y zonas de decision")
    ax.legend()
    savefig(fig, "umbrales_zonas.png")

    # trade-off latencia vs F1 (MLP vs XGB)
    data = joblib.load(BASE / "models/xgb_scaler.pkl")
    xgb = data["xgb"]
    t0 = time.perf_counter()
    for _ in range(50):
        xgb.predict_proba(Xte)
    lat_xgb = (time.perf_counter() - t0) / 50 / len(Xte) * 1000  # ms/muestra

    mlp = MLP(Xte.shape[1], "bn_then_dropout")
    mlp.load_state_dict(torch.load(BASE / "models/mlp.pt"))
    t0 = time.perf_counter()
    for _ in range(50):
        predict_mlp(mlp, Xte)
    lat_mlp = (time.perf_counter() - t0) / 50 / len(Xte) * 1000

    facts = save_facts({
        "zonas_pct": pct,
        "acc_zona_verde": round(acc_verde, 3),
        "f1_sin_filtro": round(float(f1_sin_filtro), 3),
        "f1_filtrando_rojo": round(float(f1_filtrado), 3),
        "latencia_ms_xgb": round(lat_xgb, 4),
        "latencia_ms_mlp": round(lat_mlp, 4),
    })
    assert set(np.unique(zonas)) <= {"verde", "amarillo", "rojo"}
    assert abs(sum(pct.values()) - 1.0) < 0.02
    print("zonas %:", pct, "| acc verde:", round(acc_verde, 3))
    print(f"F1 sin filtro {f1_sin_filtro:.3f} -> filtrando rojo {f1_filtrado:.3f}")
    print(f"latencia ms/muestra: XGB {lat_xgb:.4f} | MLP {lat_mlp:.4f}")
    print("OK")


if __name__ == "__main__":
    run()
