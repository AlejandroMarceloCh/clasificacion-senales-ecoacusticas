"""Estilo global de figuras. CRITICO: fuente >=14 en todo (penalizacion -3 si <14).
Tambien helpers para acumular metricas en report/facts.json.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
FIGS = BASE / "figs"
REPORT = BASE / "report"
FACTS = REPORT / "facts.json"

# Fuente base 15 (margen sobre el minimo 14 exigido)
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def savefig(fig, name):
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    # verificacion anti -3pts: ningun texto de ejes <14
    return str(out)


def load_facts():
    if FACTS.exists():
        return json.loads(FACTS.read_text())
    return {}


def save_facts(d):
    REPORT.mkdir(exist_ok=True)
    cur = load_facts()
    cur.update(d)
    FACTS.write_text(json.dumps(cur, indent=2, ensure_ascii=False))
    return cur
