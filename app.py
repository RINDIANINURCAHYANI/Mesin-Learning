import os
from flask import Flask, render_template, request
import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# =========================
# PATH BASE (AMAN DI VERCEL)
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
    # Kalau ini error, app bakal tetap hidup tapi kasih pesan jelas di halaman
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
            # Model tidak punya feature_importances_
            feature_importance_plot = None
    except Exception as e:
        # Plot gagal? gak apa-apa, app tetap jalan
        feature_importance_plot = None

def get_float(form, key):
    val = form.get(key, "")
    return float(val) if val != "" else None

def get_str(form, key):
    return (form.get(key, "") or "").strip()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = load_error  # kalau load model gagal, langsung tampilkan

    if request.method == "POST" and model is not None:
        try:
            # numerik
            age = get_float(request.form, "age")
            trestbps = get_float(request.form, "trestbps")
            chol = get_float(request.form, "chol")
            thalch = get_float(request.form, "thalch")
            oldpeak = get_float(request.form, "oldpeak")
            ca = get_float(request.form, "ca")

            # kategori
            sex = get_str(request.form, "sex")
            dataset = get_str(request.form, "dataset")
            cp = get_str(request.form, "cp")
            fbs = get_str(request.form, "fbs")
            restecg = get_str(request.form, "restecg")
            exang = get_str(request.form, "exang")
            slope = get_str(request.form, "slope")
            thal = get_str(request.form, "thal")

            # validasi minimal
            if None in [age, trestbps, chol, thalch, oldpeak, ca]:
                raise ValueError("Semua field numerik wajib diisi.")

            input_df = pd.DataFrame([{
                "age": age,
                "trestbps": trestbps,
                "chol": chol,
                "thalch": thalch,
                "oldpeak": oldpeak,
                "ca": ca,
                "sex": sex,
                "dataset": dataset,
                "cp": cp,
                "fbs": fbs,
                "restecg": restecg,
                "exang": exang,
                "slope": slope,
                "thal": thal
            }])

            # one-hot sesuai training
            if cat_cols:
                input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

            # samakan kolom input ke training
            input_df = input_df.reindex(columns=TRAIN_COLUMNS, fill_value=0)

            pred = model.predict(input_df)[0]

            # beberapa model gak punya predict_proba
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0][1]
                probability = round(proba * 100, 2)
            else:
                probability = None

            prediction = "BERISIKO Penyakit Jantung" if pred == 1 else "RISIKO RENDAH"

        except Exception as e:
            error = f"Terjadi error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
        feature_importance_plot=feature_importance_plot
    )

# Endpoint kecil buat test health di Vercel
@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    # lokal aja
    app.run(host="0.0.0.0", port=5000, debug=True)
