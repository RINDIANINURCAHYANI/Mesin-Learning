import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# load dataset
df = pd.read_csv(r"C:\Users\rindi\OneDrive\Dokumen\PENYAKIT JANTUNG\heart_disease_uci.csv")

# target biner risk
df["risk"] = (df["num"] > 0).astype(int)

# drop kolom tidak dipakai
cols_to_drop = [c for c in ["d", "id", "num"] if c in df.columns]
df = df.drop(columns=cols_to_drop)

num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
cat_cols = ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# missing values
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].mode()[0])

# one-hot
df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_model.drop(columns=["risk"])
y = df_model["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# simpan model + kolom train
joblib.dump(model, "model.pkl")
joblib.dump(list(X_train.columns), "train_columns.pkl")
joblib.dump(cat_cols, "cat_cols.pkl")

print("✅ Model & metadata tersimpan: model.pkl, train_columns.pkl, cat_cols.pkl")
