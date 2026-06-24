"""App de inferencia con escenarios de ejemplo y semaforo de umbrales."""
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

BASE = Path(__file__).resolve().parent
MEL = [f"mel_{i}" for i in range(64)]
SPECIES = {10: "Leptodactylus discodactylus (anfibio)", 12: "Osteocephalus taurinus (anfibio)",
           17: "Chiroxiphia lineata (ave)", 18: "Saltator grossus (ave)",
           23: "Pheucticus chrysopeplus (ave)"}
CLASES = [10, 12, 17, 18, 23]

st.set_page_config(page_title="Clasificador Eco-Acustico", page_icon="🐸", layout="centered")
st.title("🐦 Clasificador de Señales Eco-Acústicas")
st.caption("CS3061 · Proyecto 2 · Inferencia con política de moderación por umbrales")


@st.cache_resource
def cargar():
    data = joblib.load(BASE / "models/xgb_scaler.pkl")
    test = pd.read_csv(BASE / "data/eco_acoustic_test.csv")
    return data["xgb"], data["scaler"], test


modelo, scaler, test = cargar()

# escenarios precargados: 6 grabaciones reales del test
ids = test["recording_id"].drop_duplicates().head(6).tolist()
sel = st.selectbox("Escenario (grabación real del conjunto de prueba):", ids)
fila = test[test["recording_id"] == sel].iloc[0]

# preprocesar igual que en entrenamiento (scaler de mel + songtype one-hot)
x_mel = scaler.transform(fila[MEL].to_numpy().reshape(1, -1))
st_dummies = pd.get_dummies(test["songtype_id"], prefix="songtype").iloc[[fila.name]].to_numpy()
X = np.hstack([x_mel, st_dummies]).astype(np.float32)

proba = modelo.predict_proba(X)[0]
pred = CLASES[int(proba.argmax())]
pmax = float(proba.max())

# semaforo de 3 zonas
if pmax >= 0.85:
    zona, color, accion = "VERDE", "#2a9d8f", "Clasificación automática (alta confianza)"
elif pmax >= 0.40:
    zona, color, accion = "AMARILLA", "#e9c46a", "Derivar a auditoría humana (incertidumbre)"
else:
    zona, color, accion = "ROJA", "#e76f51", "Descartar registro (probable ruido)"

c1, c2 = st.columns([1, 1])
c1.metric("Especie predicha", f"#{pred}")
c1.write(SPECIES[pred])
c2.metric("Confianza (P máx)", f"{pmax*100:.1f}%")
st.markdown(f"<div style='background:{color};padding:14px;border-radius:8px;color:white;"
            f"font-weight:bold;font-size:18px'>Zona {zona} — {accion}</div>", unsafe_allow_html=True)

st.subheader("Distribución de probabilidad por especie")
df = pd.DataFrame({"especie": [SPECIES[c] for c in CLASES], "probabilidad": proba})
st.bar_chart(df.set_index("especie"))

st.caption("Umbrales: Verde P≥85% · Amarillo 40–85% · Rojo <40%. "
           "Especie real de este registro: #" + str(int(fila["species_id"])))
