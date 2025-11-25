import os
import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio

# =========================
# TAMBAHAN UNTUK STREAMLIT
# =========================
import streamlit as st

# =========================
# PATH BASE (AMAN DI STREAMLIT CLOUD)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {filename}")
    return joblib.load(path)

# =========================
# LOAD MODEL & METADATA
# =========================
try:
    model = load_pickle("model.pkl")
    TRAIN_COLUMNS = load_pickle("train_columns.pkl")
    cat_cols = load_pickle("cat_cols.pkl")
except Exception as e:
    model = None
    TRAIN_COLUMNS = None
    cat_cols = []
    load_error = f"Gagal load model/metadata: {e}"
else:
    load_error = None

# =========================
# OPTIONAL: FEATURE IMPORTANCE PLOT
# =========================
feature_importance_plot = None
fig_fi = None  # simpan fig Plotly biar bisa ditampilin Streamlit
if model is not None and TRAIN_COLUMNS is not None:
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            fi_df = pd.DataFrame({
                "Fitur": TRAIN_COLUMNS,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Fitur",
                orientation="h",
                title="Feature Importance",
                text="Importance"
            )
            fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_fi.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(l=140, r=40, t=60, b=40),
                height=520
            )
            feature_importance_plot = pio.to_html(fig_fi, full_html=False)
        else:
            feature_importance_plot = None
    except Exception:
        feature_importance_plot = None
        fig_fi = None

def get_float(val):
    return float(val) if val is not None and val != "" else None

def get_str(val):
    return (val or "").strip()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Prediksi Risiko Penyakit Jantung", layout="centered")
st.title("Prediksi Risiko Penyakit Jantung")

if load_error:
    st.error(load_error)
    st.stop()

st.write("Isi form di bawah untuk memprediksi risiko penyakit jantung.")

with st.form("predict_form"):
    st.subheader("Fitur Numerik")
    age = st.number_input("age", min_value=0.0, step=1.0, format="%.0f")
    trestbps = st.number_input("trestbps", min_value=0.0, step=1.0, format="%.0f")
    chol = st.number_input("chol", min_value=0.0, step=1.0, format="%.0f")
    thalch = st.number_input("thalch", min_value=0.0, step=1.0, format="%.0f")
    oldpeak = st.number_input("oldpeak", min_value=0.0, step=0.1, format="%.2f")
    ca = st.number_input("ca", min_value=0.0, step=1.0, format="%.0f")

    st.subheader("Fitur Kategori")
    sex = st.selectbox("sex", ["", "male", "female"])
    dataset = st.selectbox("dataset", ["", "cleveland", "hungarian", "switzerland", "va"])
    cp = st.selectbox("cp", ["", "typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
    fbs = st.selectbox("fbs", ["", "true", "false"])
    restecg = st.selectbox("restecg", ["", "normal", "st-t abnormality", "lv hypertrophy"])
    exang = st.selectbox("exang", ["", "yes", "no"])
    slope = st.selectbox("slope", ["", "upsloping", "flat", "downsloping"])
    thal = st.selectbox("thal", ["", "normal", "fixed defect", "reversible defect"])

    submitted = st.form_submit_button("Prediksi")

prediction = None
probability = None
error = None

if submitted and model is not None:
    try:
        # validasi minimal: numerik wajib diisi (number_input selalu ada nilainya)
        input_df = pd.DataFrame([{
            "age": get_float(age),
            "trestbps": get_float(trestbps),
            "chol": get_float(chol),
            "thalch": get_float(thalch),
            "oldpeak": get_float(oldpeak),
            "ca": get_float(ca),
            "sex": get_str(sex),
            "dataset": get_str(dataset),
            "cp": get_str(cp),
            "fbs": get_str(fbs),
            "restecg": get_str(restecg),
            "exang": get_str(exang),
            "slope": get_str(slope),
            "thal": get_str(thal)
        }])

        # one-hot sesuai training
        if cat_cols:
            input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

        # samakan kolom input ke training
        input_df = input_df.reindex(columns=TRAIN_COLUMNS, fill_value=0)

        pred = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]
            probability = round(proba * 100, 2)
        else:
            probability = None

        prediction = "BERISIKO Penyakit Jantung" if pred == 1 else "RISIKO RENDAH"

    except Exception as e:
        error = f"Terjadi error: {e}"

# =========================
# OUTPUT STREAMLIT
# =========================
if error:
    st.error(error)

if prediction is not None:
    st.success(f"Hasil Prediksi: **{prediction}**")
    if probability is not None:
        st.info(f"Probabilitas Risiko: **{probability}%**")

st.divider()

if fig_fi is not None:
    st.subheader("Feature Importance")
    st.plotly_chart(fig_fi, use_container_width=True)
