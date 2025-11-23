from flask import Flask, render_template, request
import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# =========================
# LOAD MODEL PKL
# =========================
model = joblib.load("model.pkl")
TRAIN_COLUMNS = joblib.load("train_columns.pkl")
cat_cols = joblib.load("cat_cols.pkl")

# (opsional) load data untuk bikin grafik dari feature importance
fi_df = pd.DataFrame({
    "Fitur": TRAIN_COLUMNS,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig_fi = px.bar(
    fi_df,
    x="Importance",
    y="Fitur",
    orientation="h",
    title="Feature Importance - Decision Tree",
    text="Importance"
)
fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig_fi.update_layout(
    yaxis=dict(autorange="reversed"),
    margin=dict(l=140, r=40, t=60, b=40),
    height=520
)
feature_importance_plot = pio.to_html(fig_fi, full_html=False)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            thalch = float(request.form["thalch"])
            oldpeak = float(request.form["oldpeak"])
            ca = float(request.form["ca"])

            sex = request.form["sex"]
            dataset = request.form["dataset"]
            cp = request.form["cp"]
            fbs = request.form["fbs"]
            restecg = request.form["restecg"]
            exang = request.form["exang"]
            slope = request.form["slope"]
            thal = request.form["thal"]

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

            # one-hot sama seperti training
            input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

            # samakan kolom input
            input_df = input_df.reindex(columns=TRAIN_COLUMNS, fill_value=0)

            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]

            prediction = "BERISIKO Penyakit Jantung" if pred == 1 else "RISIKO RENDAH"
            probability = round(proba * 100, 2)

        except Exception as e:
            error = f"Terjadi error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
        feature_importance_plot=feature_importance_plot
    )


if __name__ == "__main__":
    app.run(debug=True)
