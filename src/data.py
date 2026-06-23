"""Sprint 0 — Carga y preprocesamiento del dataset eco-acustico.

Decisiones de ingenieria (ver HALLAZGOS_RECON.md):
- recording_id: SE EXCLUYE  -> 26.9% de IDs de test estan en train (leakage).
- songtype_id: SE INCLUYE   -> aporta +9.4% F1 (one-hot).
- is_tp: se descarta como feature -> redundante con mel (+0.6% F1).
- mel_0..mel_63: StandardScaler fiteado SOLO en train (sin fuga al test).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

MEL_COLS = [f"mel_{i}" for i in range(64)]
SPECIES = {10: "Leptodactylus discodactylus", 12: "Osteocephalus taurinus",
           17: "Chiroxiphia lineata", 18: "Saltator grossus", 23: "Pheucticus chrysopeplus"}
TIPO = {10: "Anfibio", 12: "Anfibio", 17: "Ave", 18: "Ave", 23: "Ave"}


def load_raw():
    tr = pd.read_csv(DATA / "eco_acoustic_train.csv")
    te = pd.read_csv(DATA / "eco_acoustic_test.csv")
    return tr, te


def build_features(tr, te):
    """Devuelve X_train, X_test (escalados, con songtype one-hot), y_train, y_test, scaler."""
    # one-hot de songtype_id (alineando columnas train/test)
    st_tr = pd.get_dummies(tr["songtype_id"], prefix="songtype")
    st_te = pd.get_dummies(te["songtype_id"], prefix="songtype")
    st_te = st_te.reindex(columns=st_tr.columns, fill_value=0)

    scaler = StandardScaler().fit(tr[MEL_COLS])          # FIT solo en train
    Xtr_mel = pd.DataFrame(scaler.transform(tr[MEL_COLS]), columns=MEL_COLS, index=tr.index)
    Xte_mel = pd.DataFrame(scaler.transform(te[MEL_COLS]), columns=MEL_COLS, index=te.index)

    X_train = pd.concat([Xtr_mel, st_tr.reset_index(drop=True)], axis=1)
    X_test = pd.concat([Xte_mel, st_te.reset_index(drop=True)], axis=1)
    y_train = tr["species_id"].to_numpy()
    y_test = te["species_id"].to_numpy()
    return X_train, X_test, y_train, y_test, scaler


def get_data():
    tr, te = load_raw()
    return build_features(tr, te)


if __name__ == "__main__":
    tr, te = load_raw()
    X_train, X_test, y_train, y_test, scaler = build_features(tr, te)
    # --- TEST Sprint 0 ---
    assert X_train.shape[0] == 1906 and X_test.shape[0] == 477, "filas mal"
    assert "recording_id" not in X_train.columns, "recording_id no excluido (leakage)"
    assert abs(X_train[MEL_COLS].to_numpy().mean()) < 1e-6, "scaler no aplicado"
    assert scaler.n_samples_seen_ == 1906, "scaler vio el test (fuga)"
    assert set(np.unique(y_train)) == {10, 12, 17, 18, 23}, "clases mal"
    print(f"X_train {X_train.shape} | X_test {X_test.shape} | features={X_train.shape[1]}")
    print("balance train:", dict(pd.Series(y_train).value_counts().sort_index()))
    print("OK Sprint 0")
