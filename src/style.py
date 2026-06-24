"""Estilo de figuras y utilidades para guardar metricas."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
FIGS = BASE / "figs"
REPORT = BASE / "report"
FACTS = REPORT / "facts.json"

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
