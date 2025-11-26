import os, json
import numpy as np
from flask import Flask, request, render_template
import plotly.express as px
import plotly.io as pio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)

def load_json(filename):
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {filename}")
    with open(path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    import joblib  # hanya dipakai lokal; Vercel masih bisa load ini kalau joblib ada
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {filename}")
    return joblib.load(path)

# Load metadata (kolom & cat col) tetap dari pkl kecil
try:
    TRAIN_COLUMNS = load_pickle("train_columns.pkl")
    cat_cols = load_pickle("cat_cols.pkl") or []
    if not isinstance(cat_cols, list):
        cat_cols = list(cat_cols)

    tree_export = load_json("tree_model.json")
    CLASSES = tree_export["classes"]
    TREE = tree_export["tree"]
    load_error = None
except Exception as e:
    TRAIN_COLUMNS, cat_cols, TREE, CLASSES = None, [], None, []
    load_error = f"Gagal load model/metadata: {e}"

def get_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

def vectorize_input(form_dict):
    x = {col: 0 for col in TRAIN_COLUMNS}

    numeric_cols = ["age","trestbps","chol","thalch","oldpeak","ca"]
    for c in numeric_cols:
        x[c] = get_float(form_dict.get(c))

    for base_col in cat_cols:
        val = (form_dict.get(base_col) or "").strip()
        for tc in TRAIN_COLUMNS:
            prefix = base_col + "_"
            if tc.startswith(prefix):
                cat_val = tc[len(prefix):]
                if val == cat_val:
                    x[tc] = 1

    return np.array([x[c] for c in TRAIN_COLUMNS])

def tree_predict_proba(x_vec, node):
    if node["type"] == "leaf":
        counts = node["value"]
        total = sum(counts)
        probs = [c/total for c in counts]
        return probs

    feat = node["feature"]
    thr = node["threshold"]
    idx = TRAIN_COLUMNS.index(feat)
    if x_vec[idx] <= thr:
        return tree_predict_proba(x_vec, node["left"])
    else:
        return tree_predict_proba(x_vec, node["right"])

def make_feature_importance_plot():
    # karena kita lepas sklearn di runtime, kita nggak bisa ambil feature_importances_
    # jadi tampilkan chart kosong/placeholder biar UI nggak rusak
    # (UI kamu tetap sama, cuma chart isinya netral)
    fi_df = {"Fitur": TRAIN_COLUMNS, "Importance": [0]*len(TRAIN_COLUMNS)}
    fig = px.bar(fi_df, x="Importance", y="Fitur", orientation="h",
                 title="Feature Importance")
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(l=140, r=40, t=60, b=40),
        height=520
    )
    return fig

FIG_FI = make_feature_importance_plot() if TRAIN_COLUMNS else None

@app.route("/", methods=["GET","POST"])
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
            x_vec = vectorize_input(form)
            probs = tree_predict_proba(x_vec, TREE)

            # asumsi kelas positif = 1 ada di index 1
            proba_pos = probs[1] if len(probs) > 1 else probs[0]
            probability = round(proba_pos * 100, 2)
            pred = 1 if proba_pos >= 0.5 else 0

            prediction = "BERISIKO Penyakit Jantung" if pred == 1 else "RISIKO RENDAH"

        except Exception as e:
            error = f"Terjadi error: {e}"

    if FIG_FI is not None:
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
