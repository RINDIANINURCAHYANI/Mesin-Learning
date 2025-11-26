import os
import joblib
import numpy as np
from flask import Flask, request, render_template
import plotly.express as px
import plotly.io as pio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # balik ke root project

# Flask harus diarahkan ke templates & static di root
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)

def load_pickle(filename):
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {filename}")
    return joblib.load(path)

# Load model & metadata
try:
    model = load_pickle("model.pkl")
    TRAIN_COLUMNS = load_pickle("train_columns.pkl")
    cat_cols = load_pickle("cat_cols.pkl") or []
    if not isinstance(cat_cols, list):
        cat_cols = list(cat_cols)

    load_error = None
except Exception as e:
    model = None
    TRAIN_COLUMNS = None
    cat_cols = []
    load_error = f"Gagal load model/metadata: {e}"

def get_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

def vectorize_input(form_dict):
    """
    Bikin vektor input sesuai TRAIN_COLUMNS TANPA pandas biar ringan.
    """
    x = {col: 0 for col in TRAIN_COLUMNS}

    # numerik
    numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    for c in numeric_cols:
        x[c] = get_float(form_dict.get(c))

    # kategorikal -> one hot berdasar TRAIN_COLUMNS
    for base_col in cat_cols:
        val = (form_dict.get(base_col) or "").strip()
        for tc in TRAIN_COLUMNS:
            prefix = base_col + "_"
            if tc.startswith(prefix):
                cat_val = tc[len(prefix):]
                if val == cat_val:
                    x[tc] = 1

    return np.array([[x[c] for c in TRAIN_COLUMNS]])

def make_feature_importance_plot():
    if model is None or TRAIN_COLUMNS is None:
        return None
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None

    fi_df = {
        "Fitur": TRAIN_COLUMNS,
        "Importance": importances
    }
    fig = px.bar(
        fi_df,
        x="Importance",
        y="Fitur",
        orientation="h",
        title="Feature Importance",
        text="Importance"
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(l=140, r=40, t=60, b=40),
        height=520
    )
    return fig

FIG_FI = make_feature_importance_plot()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None
    feature_importance_plot = None

    if load_error:
        return render_template("index.html", error=load_error)

    if request.method == "POST":
        try:
            form = request.form.to_dict()
            X = vectorize_input(form)

            pred = model.predict(X)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0][1]
                probability = round(proba * 100, 2)

            prediction = "BERISIKO Penyakit Jantung" if pred == 1 else "RISIKO RENDAH"

        except Exception as e:
            error = f"Terjadi error: {e}"

    if FIG_FI is not None:
        # inject plotly div (tanpa include js biar ringan)
        feature_importance_plot = pio.to_html(
            FIG_FI, full_html=False, include_plotlyjs=False
        )

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
        feature_importance_plot=feature_importance_plot
    )
