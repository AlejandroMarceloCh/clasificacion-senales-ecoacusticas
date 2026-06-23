"""Sprint 4 — Clasificacion (C4): MLP (PyTorch) vs XGBoost.
Incluye: F1-macro con StratifiedKFold, matrices de confusion, experimento de
POSICION de Dropout/BatchNorm, evaluacion en el test.csv oficial, loss formal.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from style import savefig, save_facts
from data import get_data

torch.manual_seed(42)
np.random.seed(42)
CLASES = [10, 12, 17, 18, 23]


class MLP(nn.Module):
    """64->256->128->64->5. variante controla la posicion de Dropout/BatchNorm."""
    def __init__(self, n_in, variante="bn_then_dropout"):
        super().__init__()
        dims = [n_in, 256, 128, 64]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            if variante == "bn_then_dropout":
                layers += [nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(0.3)]
            elif variante == "dropout_then_bn":
                layers += [nn.ReLU(), nn.Dropout(0.3), nn.BatchNorm1d(b)]
            else:  # sin_reg
                layers += [nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def y_to_idx(y):
    idx = {c: i for i, c in enumerate(CLASES)}
    return np.array([idx[v] for v in y])


def train_mlp(Xtr, ytr, variante, weights, epochs=120, return_curve=False):
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    model = MLP(Xtr.shape[1], variante)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    curve = []
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward(); opt.step()
        curve.append(float(loss))
    if return_curve:
        return model, curve
    return model


def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        proba = torch.softmax(logits, 1).numpy()
    return proba


def run():
    X_train, X_test, y_train, y_test, scaler = get_data()
    Xtr, Xte = X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32)
    ytr_i, yte_i = y_to_idx(y_train), y_to_idx(y_test)
    weights = compute_class_weight("balanced", classes=np.arange(5), y=ytr_i)

    # ===== CV StratifiedKFold (F1-macro) =====
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_mlp_cv, f1_xgb_cv = [], []
    for tr_i, va_i in skf.split(Xtr, ytr_i):
        m = train_mlp(Xtr[tr_i], ytr_i[tr_i], "bn_then_dropout", weights, epochs=100)
        pred = predict_mlp(m, Xtr[va_i]).argmax(1)
        f1_mlp_cv.append(f1_score(ytr_i[va_i], pred, average="macro"))
        sw = weights[ytr_i[tr_i]]
        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                            subsample=0.9, eval_metric="mlogloss", random_state=42)
        xgb.fit(Xtr[tr_i], ytr_i[tr_i], sample_weight=sw)
        f1_xgb_cv.append(f1_score(ytr_i[va_i], xgb.predict(Xtr[va_i]), average="macro"))

    # ===== Modelos finales + evaluacion en TEST OFICIAL (477) =====
    mlp = train_mlp(Xtr, ytr_i, "bn_then_dropout", weights, epochs=140)
    proba_mlp = predict_mlp(mlp, Xte)
    pred_mlp = proba_mlp.argmax(1)
    f1_mlp_test = f1_score(yte_i, pred_mlp, average="macro")

    sw = weights[ytr_i]
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                        subsample=0.9, eval_metric="mlogloss", random_state=42)
    t0 = time.perf_counter(); xgb.fit(Xtr, ytr_i, sample_weight=sw)
    proba_xgb = xgb.predict_proba(Xte)
    pred_xgb = proba_xgb.argmax(1)
    f1_xgb_test = f1_score(yte_i, pred_xgb, average="macro")

    # matrices de confusion 5x5
    for nom, pred in [("mlp", pred_mlp), ("xgb", pred_xgb)]:
        cm = confusion_matrix(yte_i, pred)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(CLASES); ax.set_yticklabels(CLASES)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusion - {nom.upper()} (test oficial)")
        fig.colorbar(im, ax=ax).ax.tick_params(labelsize=14)
        savefig(fig, f"confusion_{nom}.png")

    # ===== Experimento POSICION Dropout/BatchNorm (curvas Loss/epoca) =====
    fig, ax = plt.subplots(figsize=(9, 5))
    curvas = {}
    for var, color in [("bn_then_dropout", "#2a9d8f"), ("dropout_then_bn", "#e76f51"),
                       ("sin_reg", "#264653")]:
        _, curve = train_mlp(Xtr, ytr_i, var, weights, epochs=140, return_curve=True)
        curvas[var] = curve
        ax.plot(curve, lw=2, label=var.replace("_", " "), color=color)
    ax.set_xlabel("Epoca"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Impacto de la posicion de Dropout/BatchNorm")
    ax.legend()
    savefig(fig, "curvas_dropout_bn.png")

    # topologia como tabla LaTeX
    topo = r"""\begin{tabular}{llll}
\toprule
Capa & Salida & Activacion & Regularizacion \\
\midrule
Linear & 256 & ReLU & BatchNorm + Dropout(0.3) \\
Linear & 128 & ReLU & BatchNorm + Dropout(0.3) \\
Linear & 64  & ReLU & BatchNorm + Dropout(0.3) \\
Linear & 5   & Softmax & --- \\
\bottomrule
\end{tabular}"""
    (save_facts.__globals__["REPORT"] / "topologia_tabla.tex").write_text(topo)

    # serializar para Streamlit/umbrales
    import joblib
    torch.save(mlp.state_dict(), save_facts.__globals__["BASE"] / "models/mlp.pt")
    joblib.dump({"xgb": xgb, "scaler": scaler}, save_facts.__globals__["BASE"] / "models/xgb_scaler.pkl")
    np.save(save_facts.__globals__["BASE"] / "models/proba_xgb_test.npy", proba_xgb)

    facts = save_facts({
        "f1_macro_mlp_cv": [round(x, 3) for x in f1_mlp_cv],
        "f1_macro_xgb_cv": [round(x, 3) for x in f1_xgb_cv],
        "f1_macro_mlp_cv_mean": round(float(np.mean(f1_mlp_cv)), 3),
        "f1_macro_xgb_cv_mean": round(float(np.mean(f1_xgb_cv)), 3),
        "f1_macro_mlp_test": round(float(f1_mlp_test), 3),
        "f1_macro_xgb_test": round(float(f1_xgb_test), 3),
        "f1_macro_test_oficial": round(float(max(f1_mlp_test, f1_xgb_test)), 3),
        "loss_formula_tex": r"\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{5} w_c\, y_{i,c}\log \hat{p}_{i,c}",
        "topologia": "64->256->128->64->5 (ReLU, BatchNorm, Dropout 0.3)",
        "curvas_dropout_bn_configs": list(curvas.keys()),
        "loss_final_por_config": {k: round(v[-1], 3) for k, v in curvas.items()},
    })
    # --- TEST Sprint 4 ---
    assert 0.30 <= f1_mlp_test <= 0.65, f"f1 mlp fuera de rango: {f1_mlp_test}"
    assert len(curvas) >= 3, "experimento dropout/bn incompleto"
    assert "f1_macro_test_oficial" in facts
    print(f"MLP  CV={np.mean(f1_mlp_cv):.3f}  test={f1_mlp_test:.3f}")
    print(f"XGB  CV={np.mean(f1_xgb_cv):.3f}  test={f1_xgb_test:.3f}")
    print("loss final por config:", facts["loss_final_por_config"])
    print("OK Sprint 4")


if __name__ == "__main__":
    run()
